#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler
import re

# --- extraction du nombre final ---
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

def extract_final_number(text: str) -> float | None:
    """Extrait le nombre après #### sinon None"""
    if not text:
        return None
    m = RE_HASH.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

# ============ CONFIG ============

class GRPOConfig:
    def __init__(
        self,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        gamma=1.0,
        clip_range=0.2,           # PPO clipping
        kl_coef=0.1,              # coefficient KL
        kl_target=None,           # seuil cible (optionnel pour early stop)
        entropy_coef=0.001,       # 🔽 bonus entropie réduit
        normalize_rewards=True,   # normalisation reward par batch
        save_steps=50,            # checkpoints périodiques
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.entropy_coef = entropy_coef
        self.normalize_rewards = normalize_rewards
        self.save_steps = save_steps


# ============ TRAINER ============

class GRPOTrainerWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        output_dir="outputs/grpo-granite",
        max_steps=100,
        temperature=0.7,
        new_tokens=256,
        writer: SummaryWriter = None,
        config: GRPOConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.temperature = temperature
        self.new_tokens = new_tokens
        self.writer = writer
        self.global_step = 0
        self.config = config or GRPOConfig()

        # Optimizer + Scheduler
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_scheduler(
            "linear",
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=max_steps,
        )

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calcule les avantages centrés/rééchelonnés pour PPO"""
        if self.config.normalize_rewards and rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards

    def kl_penalty(self, logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
        """KL divergence entre la policy courante et une policy de référence"""
        kl = (logprobs - ref_logprobs).mean()
        return self.config.kl_coef * kl

    def _save_checkpoint(self, step: int):
        """Sauvegarde périodique du modèle"""
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        print(f"💾 Checkpoint sauvegardé: {ckpt_path}")

    def _generate(self, **inputs):
        """Génération robuste avec fallback greedy"""
        try:
            return self.model.generate(
                **inputs,
                max_new_tokens=self.new_tokens,
                do_sample=True,  # on garde sampling
                temperature=self.temperature,
                top_p=0.9,
            )
        except Exception as e:
            print(f"⚠️ Sampling a échoué ({e}), fallback en greedy")
            return self.model.generate(
                **inputs,
                max_new_tokens=self.new_tokens,
                do_sample=False
            )

    def train(self, dataloader: DataLoader):
        self.model.train()
        device = next(self.model.parameters()).device

        for step, batch in enumerate(dataloader):
            if step >= self.max_steps:
                break

            prompts = batch["prompts"]
            answers = batch["answers"]

            # --- Tokenize prompts ---
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            # --- Génération ---
            with torch.no_grad():
                outputs = self._generate(**inputs)
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # --- Rewards (shaping) ---
            rewards = []
            rewards_exact, rewards_format = 0, 0
            for gen, ans in zip(generations, answers):
                pred = extract_final_number(gen)
                gold = extract_final_number(ans)

                r = 0.0
                if pred is not None:
                    rewards_format += 1
                    r += 0.2  # format respecté

                if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
                    rewards_exact += 1
                    r += 0.5  # exact match

                rewards.append(r)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantages = self.compute_advantages(rewards)

            # --- Tokenize outputs ---
            batch_outputs = self.tokenizer(generations, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = self.model(**batch_outputs).logits
            logprobs = F.log_softmax(logits, dim=-1)

            gen_logprobs = logprobs[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            ref_logprobs = gen_logprobs.detach()

            # --- PPO surrogate loss ---
            ratio = torch.exp(gen_logprobs - ref_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # --- KL penalty ---
            kl_loss = self.kl_penalty(gen_logprobs, ref_logprobs)

            # --- Entropy bonus ---
            entropy = -(logprobs.exp() * logprobs).sum(-1).mean()
            entropy_bonus = -self.config.entropy_coef * entropy

            # --- Final loss ---
            loss = policy_loss + kl_loss + entropy_bonus

            # --- Optim step ---
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # --- Logs ---
            if self.writer:
                self.writer.add_scalar("loss/policy", policy_loss.item(), self.global_step)
                self.writer.add_scalar("loss/kl", kl_loss.item(), self.global_step)
                self.writer.add_scalar("loss/entropy", entropy.item(), self.global_step)
                self.writer.add_scalar("loss/total", loss.item(), self.global_step)
                self.writer.add_scalar("reward/mean", rewards.mean().item(), self.global_step)
                self.writer.add_scalar("reward/exact", rewards_exact / max(1, len(rewards)), self.global_step)
                self.writer.add_scalar("reward/format", rewards_format / max(1, len(rewards)), self.global_step)
                self.writer.add_histogram("reward/dist", rewards.cpu().numpy(), self.global_step)

            if step % 10 == 0:
                print(
                    f"[Step {step}] loss={loss.item():.4f} reward_mean={rewards.mean().item():.3f} "
                    f"exact={rewards_exact}/{len(rewards)} format={rewards_format}/{len(rewards)} "
                    f"clip_ratio_mean={ratio.mean().item():.3f} entropy={entropy.item():.3f}"
                )

            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✅ Entraînement terminé. Modèle et tokenizer sauvegardés dans {self.output_dir}")
