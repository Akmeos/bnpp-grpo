#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRPO Trainer robuste
# - Rewards normalisées safe
# - PPO clip
# - KL control avec ref_model gelé
# - Entropy bonus
# - Logits clampés (évite NaN)
# - pad_token_id défini
# - Fallback greedy si sampling plante
# - Checkpoints périodiques

import os
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler


class GRPOConfig:
    def __init__(
        self,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        gamma=1.0,
        clip_range=0.2,
        kl_coef=0.1,
        kl_target=None,
        entropy_coef=0.01,
        normalize_rewards=True,
        save_steps=50,
        do_sample=False,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        enable_greedy_fallback=True,
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
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.enable_greedy_fallback = enable_greedy_fallback


class GRPOTrainerWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        output_dir="outputs/grpo-granite",
        max_steps=100,
        temperature=0.7,
        new_tokens=256,
        no_vllm=True,
        writer: Optional[SummaryWriter] = None,
        config: Optional[GRPOConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.new_tokens = new_tokens
        self.no_vllm = no_vllm
        self.writer = writer
        self.global_step = 0
        self.config = config or GRPOConfig(temperature=temperature)

        # Modèle de référence (gelé)
        self.ref_model = copy.deepcopy(model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

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

    # --------- utils ---------

    @staticmethod
    def _std_safe(x: torch.Tensor) -> float:
        if x.numel() <= 1:
            return 0.0
        s = x.std()
        return float(s.item()) if torch.isfinite(s) else 0.0

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_rewards:
            if rewards.numel() <= 1:
                return rewards - rewards.mean()
            std = rewards.std()
            if not torch.isfinite(std) or std < 1e-6:
                return rewards - rewards.mean()
            return (rewards - rewards.mean()) / (std + 1e-8)
        return rewards

    def kl_penalty(self, logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
        kl = (logprobs - ref_logprobs).mean()
        return self.config.kl_coef * kl

    def _save_checkpoint(self, step: int):
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        print(f"💾 Checkpoint sauvegardé: {ckpt_path}")

    def _generate(self, **inputs):
        """Génération robuste: sampling par défaut, fallback greedy sur ANY error."""
        try:
            if self.config.do_sample:
                return self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens,
                    do_sample=True,
                    temperature=max(self.config.temperature, 1e-5),
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                return self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as e:
            # Peu importe le message (y compris "device-side assert"), on bascule en greedy
            print(f"⚠️ Sampling a échoué ({e}). Fallback greedy do_sample=False.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return self.model.generate(
                **inputs,
                max_new_tokens=self.new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )


    # --------- training ---------

    def train(self, dataloader: DataLoader):
        self.model.train()
        device = next(self.model.parameters()).device

        for step, batch in enumerate(dataloader):
            if step >= self.max_steps:
                break

            prompts = batch["prompts"]
            answers = batch["answers"]

            # Prompts
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            # Génération
            with torch.no_grad():
                outputs = self._generate(**inputs)
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Rewards
            rewards = []
            for gen, ans in zip(generations, answers):
                rewards.append(1.0 if (ans and (ans in gen)) else 0.0)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Avantages
            advantages = self.compute_advantages(rewards)

            # Logprobs modèle courant
            batch_outputs = self.tokenizer(generations, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = self.model(**batch_outputs).logits.float()
            logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.clamp(logprobs, min=-20, max=0)  # ✅ clamp pour éviter -inf/+nan
            gen_logprobs = logprobs[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            # Ref logprobs
            with torch.no_grad():
                ref_logits = self.ref_model(**batch_outputs).logits.float()
                ref_logprobs_all = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs_all = torch.clamp(ref_logprobs_all, min=-20, max=0)
            ref_logprobs = ref_logprobs_all[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            # PPO
            ratio = torch.exp(gen_logprobs - ref_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # KL + Entropie
            kl_loss = self.kl_penalty(gen_logprobs, ref_logprobs)
            entropy = -(logprobs.exp() * logprobs).sum(-1).mean()
            entropy_bonus = -self.config.entropy_coef * entropy
            loss = policy_loss + kl_loss + entropy_bonus

            # Optim
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Logs
            if self.writer:
                self.writer.add_scalar("loss/policy", float(policy_loss.item()), self.global_step)
                self.writer.add_scalar("loss/kl", float(kl_loss.item()), self.global_step)
                self.writer.add_scalar("loss/entropy", float(entropy.item()), self.global_step)
                self.writer.add_scalar("loss/total", float(loss.item()), self.global_step)
                self.writer.add_scalar("reward/mean", float(rewards.mean().item()), self.global_step)
                self.writer.add_scalar("reward/std", self._std_safe(rewards), self.global_step)
                self.writer.add_scalar("clip/ratio_mean", float(ratio.mean().item()), self.global_step)

            if step % 10 == 0:
                print(
                    f"[Step {step}] loss={float(loss.item()):.4f} "
                    f"reward_mean={float(rewards.mean().item()):.3f} "
                    f"clip_ratio_mean={float(ratio.mean().item()):.3f} "
                    f"entropy={float(entropy.item()):.3f}"
                )

            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✅ Entraînement terminé. Modèle sauvegardé dans {self.output_dir}")
