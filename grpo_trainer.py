#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re
from dataclasses import fields, asdict
from typing import Any, Dict, List, Sequence

import torch
from trl import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

# --- extraction du nombre final ---
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

def extract_final_number(text: str) -> str | None:
    if not text:
        return None
    m = RE_HASH.search(text)
    if m:
        return m.group(1)
    # fallback: dernier nombre dans le texte
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return nums[-1] if nums else None

def _to_float(s):
    try:
        return float(s)
    except Exception:
        return None

def make_math_accuracy_reward(format_bonus: float = 0.10, missing_penalty: float = 0.02):
    """
    Renvoie une fonction de reward compatible GRPO.
    - +1.0 si nombre final exact (tolérance 0 absolue)
    - +format_bonus si '####' présent
    - -missing_penalty si '####' absent
    """
    def reward_fn(prompts: Sequence[str],
                  completions: Sequence[Sequence[str]] | Sequence[str],
                  answers: Sequence[str] | None = None,
                  **kwargs) -> List[float] | List[List[float]]:

        # Récupérer les références (réponses or)
        refs = answers or kwargs.get("references") or kwargs.get("labels")
        if refs is None:
            # si le batch ne contient pas 'answers', on renvoie 0
            refs = [None] * (len(prompts) if hasattr(prompts, "__len__") else 1)

        def score_one(pred_text: str, ref_text: str | None) -> float:
            has_hash = bool(RE_HASH.search(pred_text or ""))
            pred = extract_final_number(pred_text or "")
            ref = extract_final_number(ref_text or "") if ref_text else None
            s = 0.0
            if has_hash:
                s += format_bonus
            else:
                s -= missing_penalty
            if pred is not None and ref is not None:
                pf, rf = _to_float(pred), _to_float(ref)
                if pf is not None and rf is not None and math.isclose(pf, rf, rel_tol=0.0, abs_tol=1e-6):
                    s += 1.0
            # bornage
            return max(0.0, min(1.0, s))

        # Cas 1: liste de listes (bsz x num_generations)
        if len(prompts) and isinstance(completions[0], (list, tuple)):
            out: List[List[float]] = []
            for i, gens in enumerate(completions):  # chaque prompt
                ref_i = refs[i] if i < len(refs) else None
                out.append([score_one(gen, ref_i) for gen in gens])
            return out
        # Cas 2: liste plate
        else:
            out2: List[float] = []
            bsz = len(prompts)
            # si on arrive ici, on tente d'inférer num_generations pour associer les refs
            ng = max(1, len(completions) // max(1, bsz))
            for j, gen in enumerate(completions):
                ref_j = refs[j // ng] if bsz > 0 else None
                out2.append(score_one(gen, ref_j))
            return out2

    return reward_fn


class GRPOTrainerWrapper:
    """
    Enveloppe légère autour de TRL.GRPOTrainer:
      - construit un GRPOConfig propre
      - enregistre une reward "math_accuracy_reward"
      - expose train() / save_model()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        output_dir: str,
        seed: int = 42,
        use_bf16: bool = False,
        use_fp16: bool = False,
        use_vllm: bool = False,
        vllm_mode: str = "colocate",
        vllm_gpu_memory_utilization: float = 0.40,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.05,
        weight_decay: float = 0.0,
        logging_steps: int = 10,
        save_steps: int = 500,
        save_total_limit: int = 2,
        report_to: str | None = None,
        dataloader_num_workers: int = 0,
        max_prompt_length: int = 256,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 1.0,
        num_generations: int = 2,
        generation_batch_size: int = 2,
        max_steps: int = -1,
        disable_tqdm: bool = False,
        reward_format_bonus: float = 0.10,
        reward_missing_penalty: float = 0.02,
    ):
        # Construire un GRPOConfig en filtrant les champs valides
        cfg_kwargs: Dict[str, Any] = dict(
            output_dir=output_dir,
            seed=seed,
            bf16=use_bf16,
            fp16=use_fp16,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            dataloader_num_workers=dataloader_num_workers,
            report_to=report_to,
            disable_tqdm=disable_tqdm,
            # Génération
            num_generations=num_generations,
            generation_batch_size=generation_batch_size,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_new_tokens,  # TRL utilise ce nom
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature is not None and temperature > 0.0),
            # vLLM
            vllm=use_vllm,
            vllm_mode=vllm_mode,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        valid_fields = {f.name for f in fields(GRPOConfig)}
        safe_kwargs = {k: v for k, v in cfg_kwargs.items() if k in valid_fields}
        self.grpo_config = GRPOConfig(**safe_kwargs)

        # reward(s)
        reward_fn = make_math_accuracy_reward(
            format_bonus=reward_format_bonus,
            missing_penalty=reward_missing_penalty,
        )
        reward_funcs = {"math_accuracy_reward": reward_fn}

        # Trainer
        self.trainer = GRPOTrainer(
            model=model,
            args=self.grpo_config,         # même objet
            train_dataset=train_dataset,
            reward_funcs=reward_funcs,
        )

    def train(self):
        return self.trainer.train()

    def save_model(self):
        self.trainer.save_model()
