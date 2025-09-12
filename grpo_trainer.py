#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Implémentation GRPO (Grouped Reinforcement Policy Optimization)
# avec normalisation reward, clipping PPO, KL control, entropie

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any


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
        entropy_coef=0.01,        # bonus entropie
        normalize_rewards=True,   # normalisation reward par batch
        save_steps=50,            # ✅ fréquence de sauvegarde des checkpoints
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
        no_vllm=True,
        writer: SummaryWriter = None,
        config: GRPOConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.temperature = temperature
        self.new_tokens = new_tokens
        self.no_vllm = no_vllm
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
        if self.config.normalize_rewards:
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

    def train(self, dataloader: DataLoader):
        self.model.train()
        device = next(self.model.parameters()).device

        for step, batch in enumerate(dataloader):
            if step >= self.max_steps:
                break

            prompts = batch["prompts"]
            answers = batch["answers"]

            # --- Tokenize les prompts ---
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            # --- Génération avec sampling ---
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                )
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # --- Reward fonction (math accuracy simplifiée) ---
            rewards = []
            for gen, ans in zip(generations, answers):
                if ans and ans in gen:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # --- Normalisation / Avantages ---
            advantages = self.compute_advantages(rewards)

            # --- Re-tokenize les generations pour logprobs ---
            batch_outputs = self.tokenizer(generations, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = self.model(**batch_outputs).logits
            logprobs = F.log_softmax(logits, dim=-1)

            # Proxy: dernier token
            gen_logprobs = logprobs[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            # Dummy ref logprobs
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
                self.writer.add_scalar("reward/std", rewards.std().item(), self.global_step)
                self.writer.add_scalar("clip/ratio_mean", ratio.mean().item(), self.global_step)

            if step % 10 == 0:
                print(
                    f"[Step {step}] loss={loss.item():.4f} reward_mean={rewards.mean().item():.3f} "
                    f"clip_ratio_mean={ratio.mean().item():.3f} entropy={entropy.item():.3f}"
                )

            # --- ✅ Checkpoint périodique ---
            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)

        # Sauvegarde finale
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✅ Entraînement terminé. Modèle sauvegardé dans {self.output_dir}")
