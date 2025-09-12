#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRPO "serré": rewards normalisées robustes, PPO clip, KL control avec ref model,
# entropy bonus, checkpoints périodiques, génération robuste (fallback greedy),
# et calcul des logprobs en float32 pour éviter les NaN en fp16.

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
        clip_range=0.2,           # PPO clipping
        kl_coef=0.1,              # coefficient KL
        kl_target=None,           # seuil cible (optionnel pour early stop)
        entropy_coef=0.01,        # bonus entropie
        normalize_rewards=True,   # normalisation reward par batch
        save_steps=50,            # fréquence checkpoints
        # sampling par défaut (plus stable que sampling "libre")
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        # fallback si sampling casse (CUDA multinomial assert)
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
        # si un temperature est passé par train.py, on surcouche la config
        self.config = config or GRPOConfig(temperature=temperature)

        # Modèle de référence (gelé) pour le KL
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

    # --------- utilitaires ---------

    @staticmethod
    def _std_safe(x: torch.Tensor) -> float:
        if x.numel() <= 1:
            return 0.0
        s = x.std()
        return float(s.item()) if torch.isfinite(s) else 0.0

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalisation robuste des rewards"""
        if self.config.normalize_rewards:
            if rewards.numel() <= 1:
                return rewards - rewards.mean()
            std = rewards.std()
            if not torch.isfinite(std) or std < 1e-6:
                return rewards - rewards.mean()
            return (rewards - rewards.mean()) / (std + 1e-8)
        return rewards

    def kl_penalty(self, logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
        """KL approx (diff de logprobs moyens)"""
        kl = (logprobs - ref_logprobs).mean()
        return self.config.kl_coef * kl

    def _save_checkpoint(self, step: int):
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        print(f"💾 Checkpoint sauvegardé: {ckpt_path}")

    def _generate(self, **inputs):
        """Génération robuste: sampling par défaut, fallback greedy si problème."""
        try:
            return self.model.generate(
                **inputs,
                max_new_tokens=self.new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
        except RuntimeError as e:
            if self.config.enable_greedy_fallback:
                print("⚠️ Sampling a échoué, fallback en greedy (do_sample=False).", e)
                torch.cuda.empty_cache()
                return self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens,
                    do_sample=False,   # greedy
                )
            raise

    # --------- boucle d'entraînement ---------

    def train(self, dataloader: DataLoader):
        self.model.train()
        device = next(self.model.parameters()).device

        for step, batch in enumerate(dataloader):
            if step >= self.max_steps:
                break

            prompts = batch["prompts"]
            answers = batch["answers"]

            # Tokenize prompts
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            # Génération (robuste)
            with torch.no_grad():
                outputs = self._generate(**inputs)
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Reward (math accuracy "contient la réponse")
            rewards = []
            for gen, ans in zip(generations, answers):
                rewards.append(1.0 if (ans and (ans in gen)) else 0.0)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Avantages
            advantages = self.compute_advantages(rewards)

            # Re-tokenize generations
            batch_outputs = self.tokenizer(generations, return_tensors="pt", padding=True, truncation=True).to(device)

            # IMPORTANT: calcule en float32 pour stabilité
            logits = self.model(**batch_outputs).logits.float()
            logprobs = F.log_softmax(logits, dim=-1)
            gen_logprobs = logprobs[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            with torch.no_grad():
                ref_logits = self.ref_model(**batch_outputs).logits.float()
                ref_logprobs_all = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = ref_logprobs_all[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            # PPO surrogate
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

            # Logs TB
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

        # Sauvegarde finale
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✅ Entraînement terminé. Modèle sauvegardé dans {self.output_dir}")
