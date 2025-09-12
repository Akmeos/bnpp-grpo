#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRPO Trainer robuste :
# - Découpe suffixe au bon offset partagé (len(input_ids))
# - LogitsProcessor: boost des digits sur 1ers pas du suffixe
# - bad_words_ids: interdit "Q:" / "A:" au début du suffixe
# - do_sample + min_new_tokens=2 pour éviter suffixe vide
# - Reward shaping progressif + filet 0.05 si nb n'importe où
# - PPO clip + KL (ref gelé) + entropie basse
# - clamp/nan_to_num, fallback greedy, logs TB + debug batch 0

import os
import copy
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor

# === Regex util ===
RE_BEGIN_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")       # nombre AU DÉBUT du suffixe
RE_HASH      = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")    # nombre label "#### n"
RE_ANY       = re.compile(r"-?\d+(?:\.\d+)?")             # n'importe quel nombre


def extract_label_number(text: str) -> float | None:
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
    if not suffix:
        return None
    m = RE_BEGIN_NUM.match(suffix.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


class DigitPrefixBoost(LogitsProcessor):
    """Biaise les ids {0-9, '.', '-'} pendant les 1ers pas du suffixe."""
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
    """Interdit uniquement les séquences 'Q:' et 'A:' (en débuts de tokens), pas la lettre 'Q' en général."""
    bad_phrases = ["Q:", "A:", "\nQ:", "\nA:", "Question:", "Answer:"]
    bad = []
    for s in bad_phrases:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            bad.append(ids)
    return bad


class GRPOConfig:
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
        do_sample=True,          # ✅ on échantillonne pour avoir des chiffres
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

        # Modèle de ref gelé pour KL
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

        # Prépare bad_words_ids pour bannir "Q:" / "A:"
        self.bad_words_ids = build_bad_words_ids(self.tokenizer)

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

    def _generate(self, base_len: int, **inputs):
        lp = LogitsProcessorList([DigitPrefixBoost(self.tokenizer, base_len=base_len, boost=5.0, max_prefix_len=8)])
        try:
            if self.config.do_sample:
                return self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens,
                    min_new_tokens=2,                       # ✅ évite suffixe vide
                    do_sample=True,
                    temperature=max(self.config.temperature, 1e-5),
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    logits_processor=lp,
                    bad_words_ids=self.bad_words_ids or None,  # ✅ bannit "Q:"/ "A:"
                )
            else:
                return self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens,
                    min_new_tokens=2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    logits_processor=lp,
                    bad_words_ids=self.bad_words_ids or None,
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
                min_new_tokens=2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=lp,
                bad_words_ids=self.bad_words_ids or None,
            )

    # ===== Reward shaping progressif =====
    def _reward(self, suffix: str, gold_str: str) -> float:
        """
        1.00 exact ; 0.80 err_rel<1e-3 ; 0.60 err_abs<1 ou err_rel<1% ;
        0.40 err_rel<10% ; 0.20 err_rel<20% ; 0.10 nombre en tête ;
        0.05 nombre quelque part ; 0.00 sinon.
        """
        head = extract_pred_number_from_suffix_head(suffix)
        gold = extract_label_number(gold_str)

        if head is not None and gold is not None:
            err_abs = abs(head - gold)
            base = max(1.0, abs(gold))
            err_rel = err_abs / base
            if err_abs == 0:
                return 1.0
            if err_rel < 1e-3:
                return 0.8
            if err_abs < 1.0 or err_rel < 0.01:
                return 0.6
            if err_rel < 0.10:
                return 0.4
            if err_rel < 0.20:
                return 0.2
            return 0.1

        if head is not None:
            return 0.1

        if RE_ANY.search(suffix or ""):
            return 0.05

        return 0.0

    def train(self, dataloader: DataLoader):
        self.model.train()
        device = next(self.model.parameters()).device

        for step, batch in enumerate(dataloader):
            if step >= self.max_steps:
                break

            prompts = batch["prompts"]
            golds = batch["answers"]

            # Tokenize prompts
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            # ✅ longueur PARTAGÉE des entrées
            shared_in_len = inputs["input_ids"].shape[1]

            # Génération (digits boost + bad_words)
            with torch.no_grad():
                outputs = self._generate(shared_in_len, **inputs)  # [B, shared_in_len + new]

            # Suffixes = tokens après shared_in_len
            suffixes = []
            for i in range(outputs.size(0)):
                suf_ids = outputs[i, shared_in_len:] if outputs.size(1) > shared_in_len else outputs[i, 0:0]
                suffixes.append(self.tokenizer.decode(suf_ids, skip_special_tokens=True))

            # Rewards
            rewards_list = [self._reward(suf, gold) for suf, gold in zip(suffixes, golds)]
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            rewards = torch.clamp(torch.nan_to_num(rewards, nan=0.0), 0.0, 1.0)

            # Debug batch 0
            if self.config.debug_print_first_batch and step == 0:
                print("=== DEBUG batch 0 ===")
                for i in range(min(3, len(prompts))):
                    print("SUFFIX GENERATED ↓")
                    print(suffixes[i][:200].replace("\n", "\\n"))
                    print("LABEL =", golds[i])
                    print("REWARD =", rewards_list[i])
                print("=====================")

            # Skip update si tout 0
            if float(rewards.sum().item()) == 0.0:
                if self.writer:
                    self.writer.add_scalar("reward/mean", 0.0, self.global_step)
                    self.writer.add_scalar("reward/std", 0.0, self.global_step)
                if step % 10 == 0:
                    print(f"[Step {step}] skip update (all rewards=0)")
                self.global_step += 1
                continue

            advantages = self.compute_advantages(rewards)

            # Re-tokenize suffixes (policy & ref sur le même texte)
            outs = self.tokenizer(suffixes, return_tensors="pt", padding=True, truncation=True).to(device)

            # Logits courants
            logits = self.model(**outs).logits
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.clamp(torch.nan_to_num(logprobs, nan=-20.0), min=-20, max=0)
            gen_logprobs = logprobs[:, -1, :].gather(1, outs["input_ids"][:, -1].unsqueeze(-1)).squeeze()

            # Logprobs référence
            with torch.no_grad():
                ref_logits = self.ref_model(**outs).logits
                ref_logits = torch.nan_to_num(ref_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                ref_lp_all = F.log_softmax(ref_logits, dim=-1)
                ref_lp_all = torch.clamp(torch.nan_to_num(ref_lp_all, nan=-20.0), min=-20, max=0)
            ref_logprobs = ref_lp_all[:, -1, :].gather(1, outs["input_ids"][:, -1].unsqueeze(-1)).squeeze()

            # PPO
            ratio = torch.exp(torch.clamp(gen_logprobs - ref_logprobs, min=-20, max=20))
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # KL + Entropie
            kl_loss = self.kl_penalty(gen_logprobs, ref_logprobs)
            entropy = -(logprobs[:, -1, :].exp() * logprobs[:, -1, :]).sum(-1).mean()
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
