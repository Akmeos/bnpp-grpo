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
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
from transformers import GenerationConfig

# === Regex utilities for number extraction ===
RE_BEGIN_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")       # Number at BEGINNING of suffix
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")        # Number after label "#### n"
RE_ANY = re.compile(r"-?\d+(?:\.\d+)?")                  # Any number in text


def extract_label_number(text: str) -> float | None:
    """Extract the final answer number from GSM8K solution text."""
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


def extract_pred_number_from_suffix_head(suffix: str) -> float | None:
    """
    Extract the first number at the beginning of the suffix after removing "####".
    Used to parse model-generated answers.
    """
    if not suffix:
        return None
    
    # Remove "####" and surrounding whitespace
    clean_suffix = suffix.replace("####", "").strip()
    if not clean_suffix:
        return None
    
    # Take only the first word (the number)
    first_word = clean_suffix.split()[0] if clean_suffix else ""
    
    # Regex pattern for numbers with decimal point and negative sign
    num_pattern = r"^-?\d+(?:\.\d+)?"
    match = re.match(num_pattern, first_word)
    
    if match:
        try:
            return float(match.group(0))
        except (ValueError, TypeError):
            return None
    return None


class DigitPrefixBoost(LogitsProcessor):
    """Boost digit-related tokens during the first steps of suffix generation."""
    def __init__(self, tokenizer, base_len: int, boost: float = 5.0, max_prefix_len: int = 8):
        self.base_len = base_len
        self.max_prefix_len = max_prefix_len
        self.boost = boost
        allowed_chars = list("0123456789") + ['.', '-']
        allowed_ids = set()
        for ch in allowed_chars:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if len(ids) == 1:
                allowed_ids.add(ids[0])
        self.allowed_ids = list(allowed_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[1]
        gen_len = max(0, cur_len - self.base_len)
        if self.allowed_ids and gen_len < self.max_prefix_len:
            scores[:, self.allowed_ids] = scores[:, self.allowed_ids] + self.boost
        return scores


def build_bad_words_ids(tokenizer):
    """Create bad words IDs to prevent the model from generating question/answer prefixes."""
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
        kl_target=None,
        entropy_coef=0.001,
        normalize_rewards=True,
        save_steps=50,
        do_sample=True,          # Enable sampling to get diverse digits
        temperature=0.7,
        top_k=40,
        top_p=0.9,
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
    """Main GRPO trainer class implementing the training loop with RL."""
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.new_tokens = new_tokens
        self.writer = writer
        self.global_step = 0
        self.config = config or GRPOConfig(temperature=temperature)

        # Frozen reference model for KL divergence calculation
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

        # Prepare bad words IDs to ban "Q:" / "A:" prefixes
        self.bad_words_ids = build_bad_words_ids(self.tokenizer)

    @staticmethod
    def _std_safe(x: torch.Tensor) -> float:
        """Safe standard deviation calculation with edge case handling."""
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
        """Calculate KL divergence penalty between current and reference policy."""
        kl = (logprobs - ref_logprobs).mean()
        return self.config.kl_coef * kl

    def _save_checkpoint(self, step: int):
        """Save model checkpoint at specified training step."""
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        print(f" Checkpoint saved: {ckpt_path}")
    
    def _generate_simple_fallback(self, inputs):
        """Ultra-simple fallback generation without any advanced options."""
        print("Using ultra-simple fallback generation")
        try:
            return self.model.generate(
                **inputs,
                max_new_tokens=self.new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except Exception as e:
            print(f"Even simple fallback failed: {e}")
            # Last resort: return input as is
            return inputs["input_ids"]

    def _generate(self, base_len: int, **inputs):
        """Simple generation without complex logits processing."""
        try:
            # Very simple generation without forcing
            return self.model.generate(
                **inputs,
                max_new_tokens=self.new_tokens,
                do_sample=self.config.do_sample,
                temperature=max(0.5, self.config.temperature),
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        except Exception as e:
            print(f"Generation failed: {e}")
            return inputs["input_ids"]

    def _reward(self, suffix: str, gold_str: str) -> float:
        """
        Calculate reward based on answer correctness with penalty for absurd numbers.
        More permissive version that accepts numbers even without exact '####' prefix.
        """
        # Try to extract number even if format isn't perfect
        pred = extract_pred_number_from_suffix_head(suffix)
        
        # Also try to find any number in the suffix as fallback
        if pred is None:
            numbers = RE_ANY.findall(suffix)
            if numbers:
                try:
                    pred = float(numbers[0])
                except (ValueError, TypeError):
                    pred = None
        
        try:
            gold = float(gold_str) if gold_str else None
            
            if gold is None or pred is None:
                return 0.0
            
            # Penalty for unreasonable numbers
            if abs(pred) > 1000000:  # Numbers > 1 million = absurd
                return 0.01
            if abs(pred - gold) > 100000:  # Error > 100,000 = absurd
                return 0.01
                
            # Bonus for correct format
            format_bonus = 0.2 if suffix.strip().startswith("####") else 0.0
            
            # Normal reward calculation
            if pred == gold:
                return 1.0 + format_bonus
            elif abs(pred - gold) < 0.01:
                return 0.8 + format_bonus
            elif abs(pred - gold) / max(1.0, abs(gold)) < 0.1:
                return 0.4 + format_bonus
            else:
                return 0.1 + format_bonus
        except (ValueError, TypeError):
            return 0.0
        
    
    def train(self, dataloader: DataLoader):
        """Main training loop for GRPO."""
        self.model.train()
        device = next(self.model.parameters()).device

        for step, batch in enumerate(dataloader):
            if step >= self.max_steps:
                break

            prompts = batch["prompts"]
            golds = batch["answers"]

            # Tokenize prompts
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs = {k: v for k, v in inputs.items() if v is not None}

            # Shared input length for all prompts
            shared_in_len = inputs["input_ids"].shape[1]

            # Generate suffixes (with digit boost + bad words filtering)
            with torch.no_grad():
                outputs = self._generate(shared_in_len, **inputs)  # [B, shared_in_len + new]

            # Extract suffixes (tokens after shared_in_len)
            suffixes = []
            for i in range(outputs.size(0)):
                if outputs.size(1) > shared_in_len:
                    suf_ids = outputs[i, shared_in_len:]
                else:
                    suf_ids = outputs[i, 0:0]  # Empty tensor
                suffix_text = self.tokenizer.decode(suf_ids, skip_special_tokens=True)
                suffixes.append(suffix_text)
                
                # Debug: show what was actually generated
                if i == 0 and step % 2 == 0:  # Print first sample every 2 steps
                    full_output = self.tokenizer.decode(outputs[i], skip_special_tokens=False)
                    print(f"Full generated: {full_output}")
                    print(f"Extracted suffix: '{suffix_text}'")

            # Calculate rewards
            rewards_list = [self._reward(suf, gold) for suf, gold in zip(suffixes, golds)]
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            rewards = torch.clamp(torch.nan_to_num(rewards, nan=0.0), 0.0, 1.0)
            
            if step % 5 == 0:  # Print only every 5 steps
                print(f"=== DEBUG Step {step} ===")
                print(f"Suffix: '{suffixes[0]}'")
                print(f"Extracted number: {extract_pred_number_from_suffix_head(suffixes[0])}")
                print(f"Gold number: {golds[0]}")
                print(f"Reward: {rewards_list[0]}")
                print("========================")

            # Debug first batch
            if self.config.debug_print_first_batch and step == 0:
                print("=== DEBUG batch 0 ===")
                for i in range(min(3, len(prompts))):
                    print("SUFFIX GENERATED ↓")
                    print(suffixes[i][:200].replace("\n", "\\n"))
                    print("LABEL =", golds[i])
                    print("REWARD =", rewards_list[i])
                print("=====================")

            # Skip update if all rewards are zero
            if float(rewards.sum().item()) == 0.0:
                if self.writer:
                    self.writer.add_scalar("reward/mean", 0.0, self.global_step)
                    self.writer.add_scalar("reward/std", 0.0, self.global_step)
                if step % 10 == 0:
                    print(f"[Step {step}] skip update (all rewards=0)")
                self.global_step += 1
                continue

            advantages = self.compute_advantages(rewards)

            # Re-tokenize suffixes (policy & ref on same text)
            outs = self.tokenizer(suffixes, return_tensors="pt", padding=True, truncation=True).to(device)

            # Current policy logits
            logits = self.model(**outs).logits
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.clamp(torch.nan_to_num(logprobs, nan=-20.0), min=-20, max=0)
            gen_logprobs = logprobs[:, -1, :].gather(1, outs["input_ids"][:, -1].unsqueeze(-1)).squeeze()

            # Reference logprobs
            with torch.no_grad():
                ref_logits = self.ref_model(**outs).logits
                ref_logits = torch.nan_to_num(ref_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                ref_lp_all = F.log_softmax(ref_logits, dim=-1)
                ref_lp_all = torch.clamp(torch.nan_to_num(ref_lp_all, nan=-20.0), min=-20, max=0)
            ref_logprobs = ref_lp_all[:, -1, :].gather(1, outs["input_ids"][:, -1].unsqueeze(-1)).squeeze()

            # PPO loss calculation
            ratio = torch.exp(torch.clamp(gen_logprobs - ref_logprobs, min=-20, max=20))
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # KL + Entropy regularization
            kl_loss = self.kl_penalty(gen_logprobs, ref_logprobs)
            entropy = -(logprobs[:, -1, :].exp() * logprobs[:, -1, :]).sum(-1).mean()
            entropy_bonus = -self.config.entropy_coef * entropy
            loss = policy_loss + kl_loss + entropy_bonus

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Logging
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
                
            if step % 5 == 0:
                print(f"Prompt: {prompts[0][-50:]}...")
                print(f"Suffix: '{suffixes[0]}'")
                print(f"Gold: {golds[0]}")
                print(f"Pred extracted: {extract_pred_number_from_suffix_head(suffixes[0])}")
                print(f"Reward: {rewards_list[0]}")
                print("---")

            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Training completed. Model saved in {self.output_dir}")