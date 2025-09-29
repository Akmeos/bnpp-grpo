#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO Trainer for IBM Granite MoE model fine-tuning with Reinforcement Learning.
Implements Grouped Reinforcement Policy Optimization with LoRA adaptation.
Features robust training with digit prefix boosting, reward shaping, and memory optimization.
"""

import os
import copy
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler
from utils.memory import report_memory

# === Regex utilities for number extraction ===
RE_BEGIN_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")        # Number at beginning
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")         # Number after "####"
RE_ANY = re.compile(r"-?\d+(?:\.\d+)?")                   # Any number


def extract_label_number(text: str) -> float | None:
    """Extract the final gold answer number from GSM8K solution text."""
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


def build_bad_words_ids(tokenizer):
    """Ban 'Q:' / 'A:' style prefixes in generation."""
    bad_phrases = ["Q:", "A:", "\nQ:", "\nA:", "Question:", "Answer:"]
    bad = []
    for s in bad_phrases:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            bad.append(ids)
    return bad


class GRPOConfig:
    """Configuration class for GRPO training parameters."""
    def __init__(
        self,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        gamma=1.0,
        clip_range=0.2,
        kl_coef=0.05,
        entropy_coef=0.001,
        normalize_rewards=True,
        save_steps=50,
        do_sample=True,
        temperature=0.7,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.normalize_rewards = normalize_rewards
        self.save_steps = save_steps
        self.do_sample = do_sample
        self.temperature = temperature


class GRPOTrainerWrapper:
    """Main GRPO trainer implementing training loop with RL."""

    def __init__(
        self,
        model,
        tokenizer,
        output_dir="outputs/grpo-granite",
        max_steps=100,
        temperature=0.7,
        new_tokens=16,
        writer: SummaryWriter = None,
        config: GRPOConfig = None,
        reward_fn=None,  # ✅ custom reward function
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.new_tokens = new_tokens
        self.writer = writer
        self.global_step = 0
        self.config = config or GRPOConfig(temperature=temperature)
        self.reward_fn = reward_fn  # ✅ assign custom reward fn if provided

        # Frozen reference model for KL divergence
        self.ref_model = copy.deepcopy(model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer and scheduler
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

        # Bad words filter
        self.bad_words_ids = build_bad_words_ids(self.tokenizer)

    # === Utilities ===
    @staticmethod
    def _std_safe(x: torch.Tensor) -> float:
        if x.numel() <= 1:
            return 0.0
        s = x.std()
        return float(s.item()) if torch.isfinite(s) else 0.0

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards to compute advantages for policy gradient."""
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
        """KL divergence penalty between current and reference policy."""
        kl = (logprobs - ref_logprobs).mean()
        return self.config.kl_coef * kl

    # === Default reward function ===
    def _reward(self, suffix: str, gold_str: str) -> float:
        """Default reward function: compare predicted number vs gold."""
        numbers = RE_ANY.findall(suffix)
        if not numbers:
            return 0.0
        try:
            pred = float(numbers[-1])
            gold = float(gold_str) if gold_str else None
            if gold is None:
                return 0.0
            if pred == gold:
                return 1.0
            elif abs(pred - gold) < 0.1:
                return 0.6
            elif abs(pred - gold) / max(1.0, abs(gold)) < 0.2:
                return 0.3
            else:
                return 0.1
        except Exception:
            return 0.0

    # === Training loop ===
    def train(self, dataloader: DataLoader):
        self.model.train()
        device = next(self.model.parameters()).device

        for step, batch in enumerate(dataloader):
            if step >= self.max_steps:
                break

            prompts = batch["prompts"]
            golds = batch["answers"]

            # Tokenize
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            report_memory("After forward (activations)")

            # Extract suffixes
            shared_in_len = inputs["input_ids"].shape[1]
            suffixes = [
                self.tokenizer.decode(
                    outputs[i, shared_in_len:].cpu(), skip_special_tokens=True
                )
                for i in range(outputs.size(0))
            ]

            # Rewards
            rewards_list = []
            for suf, gold in zip(suffixes, golds):
                if self.reward_fn is not None:
                    reward = self.reward_fn(suf, gold)
                else:
                    reward = self._reward(suf, gold)
                rewards_list.append(reward)

            if step % 5 == 0:
                print(f"Step {step}:")
                print(f"Prompt: {prompts[0][-50:]}...")
                print(f"Generated: '{suffixes[0]}'")
                print(f"Gold: {golds[0]}")
                print(f"Reward: {rewards_list[0]}")
                print("---")

            rewards = torch.tensor(rewards_list, dtype=torch.float32).to(device)
            if rewards.sum().item() == 0.0:
                if step % 10 == 0:
                    print(f"[Step {step}] skip update (all rewards=0)")
                self.global_step += 1
                continue

            # Retokenize for policy gradient
            outs = self.tokenizer(suffixes, return_tensors="pt", padding=True, truncation=True)
            outs = {k: v.to(device) for k, v in outs.items()}

            # Logits + logprobs
            logits = self.model(**outs).logits
            logprobs = F.log_softmax(logits, dim=-1)
            gen_logprobs = logprobs[:, -1, :].gather(1, outs["input_ids"][:, -1].unsqueeze(-1)).squeeze()
            report_memory("After logits")

            # Reference logprobs
            with torch.no_grad():
                ref_logits = self.ref_model(**outs).logits
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = ref_logprobs[:, -1, :].gather(1, outs["input_ids"][:, -1].unsqueeze(-1)).squeeze()

            # PPO loss
            ratio = torch.exp(gen_logprobs - ref_logprobs)
            advantages = self.compute_advantages(rewards)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # KL + entropy
            kl_loss = self.kl_penalty(gen_logprobs, ref_logprobs)
            entropy = -(logprobs.exp() * logprobs).sum(-1).mean()
            loss = policy_loss + kl_loss - self.config.entropy_coef * entropy

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            report_memory("After backward")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            report_memory("After optimizer step")

            self.global_step += 1

            # Logging
            if self.writer:
                self.writer.add_scalar("loss/total", float(loss.item()), self.global_step)
                self.writer.add_scalar("reward/mean", float(rewards.mean().item()), self.global_step)

            if step % 10 == 0:
                print(f"[Step {step}] loss={float(loss.item()):.4f} reward_mean={float(rewards.mean().item()):.3f}")

            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)

        # Save final model
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Training completed. Model saved in {self.output_dir}")

    def _save_checkpoint(self, step: int):
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
