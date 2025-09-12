#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRPO Trainer robuste :
# - Format de sortie imposé côté data_loader (#### )
# - Extraction numérique robuste
# - Reward shaping (jamais tout à zéro si format suivi)
# - Skip update si reward==0 partout (sécurité)
# - PPO clip + KL vs ref_model gelé + Entropy bonus faible
# - clamp/nan_to_num
# - Fallback greedy si generate() plante
# - Logs TensorBoard + debug batch 0

import os
import copy
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler

# Extraction '#### n' prioritaire sinon dernier nombre
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
RE_ANY  = re.compile(r"-?\d+(?:\.\d+)?")

def extract_number_any(text: str) -> float | None:
    if not text:
        return None
    m = RE_HASH.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    nums = list(RE_ANY.finditer(text))
    if nums:
        try:
            return float(nums[-1].group(0))
        except Exception:
            return None
    return None


class GRPOConfig:
    def __init__(
        self,
        learning_rate=5e-5,      # low LR pour stabilité T4
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        gamma=1.0,
        clip_range=0.2,
        kl_coef=0.05,
        kl_target=None,
        entropy_coef=0.001,      # bonus entropie faible
        normalize_rewards=True,
        save_steps=50,
        do_sample=False,         # greedy par défaut (Kaggle)
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        enable_greedy_fallback=True,
        debug_print_first_batch=True,
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
        self.debug_print_first_batch = debug_print_first_batch


class GRPOTrainerWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        output_dir="outputs/grpo-granite",
        max_steps=100,
        temperature=0.7,
        new_tokens=32,   # court car on force '#### ' (quelques digits suffisent)
        writer: SummaryWriter = None,
        config: GRPOConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.new_tokens = new_tokens
        self.writer = writer
        self.global_step = 0
        self.config = config or GRPOConfig(temperature=temperature)

        # Modèle de référence gelé pour KL
        self.ref_model = copy.deepcopy(model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optim & scheduler
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_scheduler(
            "linear",
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=max(self.max_steps, 1),
        )

    @staticmethod
    def _std_safe(x: torch.Tensor) -> float:
        if x.numel() <= 1:
            return 0.0
        s = x.std()
        return float(s.item()) if torch.isfinite(s) else 0.0

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=0.0)
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
            print(f"⚠️ Sampling a échoué ({e}). Fallback greedy.")
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

    def _shape_reward(self, generation: str, answer: str) -> float:
        """
        Reward façonnée :
        +0.20 si '####' présent
        +0.20 si un nombre est détecté après parsing
        +0.60 si exact (diff==0)
        (clip 0..1)
        """
        score = 0.0
        if "####" in generation:
            score += 0.20

        pred = extract_number_any(generation)
        gold = extract_number_any(answer)

        if pred is not None:
            score += 0.20

        if pred is not None and gold is not None and abs(pred - gold) == 0:
            score += 0.60

        return min(max(score, 0.0), 1.0)

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

            # Génération
            with torch.no_grad():
                outputs = self._generate(**inputs)
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Rewards (shaping)
            rewards_list = [self._shape_reward(gen, ans) for gen, ans in zip(generations, answers)]
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            rewards = torch.clamp(torch.nan_to_num(rewards, nan=0.0), 0.0, 1.0)

            # Debug premier batch
            if self.config.debug_print_first_batch and step == 0:
                print("=== DEBUG batch 0 ===")
                for i in range(min(2, len(prompts))):
                    print("PROMPT (fin) ↓")
                    print("\n".join(prompts[i].splitlines()[-4:]))
                    print("GENERATION ↓")
                    print(generations[i][:240].replace("\n", "\\n"))
                    print("LABEL =", answers[i])
                    print("REWARD =", rewards_list[i])
                print("=====================")

            # Skip update si toutes les rewards sont nulles (sécurité)
            if float(rewards.sum().item()) == 0.0:
                if self.writer:
                    self.writer.add_scalar("reward/mean", 0.0, self.global_step)
                    self.writer.add_scalar("reward/std", 0.0, self.global_step)
                if step % 10 == 0:
                    print(f"[Step {step}] skip update (all rewards=0)")
                self.global_step += 1
                continue

            # Avantages
            advantages = self.compute_advantages(rewards)

            # Re-tokenize generations
            batch_outputs = self.tokenizer(generations, return_tensors="pt", padding=True, truncation=True).to(device)

            # Logits courants
            logits = self.model(**batch_outputs).logits
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.clamp(torch.nan_to_num(logprobs, nan=-20.0), min=-20, max=0)
            gen_logprobs = logprobs[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            # Logprobs référence
            with torch.no_grad():
                ref_logits = self.ref_model(**batch_outputs).logits
                ref_logits = torch.nan_to_num(ref_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                ref_lp_all = F.log_softmax(ref_logits, dim=-1)
                ref_lp_all = torch.clamp(torch.nan_to_num(ref_lp_all, nan=-20.0), min=-20, max=0)
            ref_logprobs = ref_lp_all[:, -1, :].gather(
                1, batch_outputs["input_ids"][:, -1].unsqueeze(-1)
            ).squeeze()

            # PPO
            ratio = torch.exp(torch.clamp(gen_logprobs - ref_logprobs, min=-20, max=20))
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
