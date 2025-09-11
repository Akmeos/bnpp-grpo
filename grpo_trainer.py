#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re
import inspect
from typing import List, Optional

from transformers import GenerationConfig
from trl import GRPOTrainer, GRPOConfig

# -----------------------------
# Extraction robuste du résultat
# -----------------------------
_RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_RE_FINAL = re.compile(
    r"(?:final answer|therefore[, ]*the answer|so the answer|the answer)\s*(?:is|=)?\s*[:]*\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RE_TRAIL_EQ = re.compile(r"=\s*(-?\d+(?:\.\d+)?)\s*$")
_RE_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")


def extract_numeric_answer(text: str) -> Optional[str]:
    """
    Priorité :
      1) '#### <num>' (format GSM8K)
      2) motifs '... answer is <num>'
      3) nombre après '=' en fin de ligne
      4) dernier nombre présent dans le texte
    """
    if not text:
        return None
    for rgx in (_RE_HASH, _RE_FINAL, _RE_TRAIL_EQ):
        m = rgx.search(text)
        if m:
            return m.group(1)
    nums = _RE_NUM.findall(text)
    return nums[-1] if nums else None


# -----------------------------
# Fonction de récompense (GRPO)
# -----------------------------
_FORMAT_BONUS = 0.1  # bonus léger si la complétion respecte '#### <num>'


def math_accuracy_reward(
    prompts: Optional[List[str]] = None,
    completions: Optional[List[str]] = None,
    answer: Optional[List[str]] = None,
    samples: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Compatible avec différentes versions de TRL :
    - réponses du modèle : `completions` (ou `samples`)
    - gold : paramètre `answer` (ou via kwargs)
    Renvoie une liste de rewards float par échantillon.
    """
    if completions is None:
        completions = samples
    gold = answer or kwargs.get("answer", None)

    rewards: List[float] = []
    if completions is None or gold is None:
        return [0.0 for _ in range(len(completions or []))]

    for comp, g in zip(completions, gold):
        fmt_bonus = _FORMAT_BONUS if _RE_HASH.search(comp or "") else 0.0

        pred_num = extract_numeric_answer(comp or "")
        gold_num = extract_numeric_answer(g or "")

        if pred_num is None or gold_num is None:
            rewards.append(fmt_bonus)
            continue

        try:
            ok = math.isclose(float(pred_num), float(gold_num), rel_tol=0.0, abs_tol=1e-6)
            rewards.append(1.0 + fmt_bonus if ok else fmt_bonus)
        except Exception:
            rewards.append(fmt_bonus)

    return rewards


# -----------------------------
# Utilitaires compatibilité API
# -----------------------------
def _filter_kwargs_for_class(cls, kwargs: dict) -> dict:
    """
    Ne garde que les kwargs acceptés par la signature de `cls.__init__`.
    """
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid}


def _sanitize_sampling_params(temperature: float):
    """
    Si temperature <= 0, on désactive le sampling :
      - do_sample = False
      - temperature ignorée
    Sinon, do_sample = True.
    """
    if temperature is None or temperature <= 0.0:
        return None, False
    return float(temperature), True


# -----------------------------
# Wrapper GRPOTrainer
# -----------------------------
class GRPOTrainerWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        output_dir: str,
        seed: int = 42,
        use_bf16: bool = False,
        use_fp16: bool = False,
        use_vllm: bool = False,
        vllm_mode: str = "colocate",
        vllm_gpu_memory_utilization: float = 0.5,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.05,
        weight_decay: float = 0.0,
        logging_steps: int = 10,
        save_steps: int = 100,
        save_total_limit: int = 2,
        report_to: Optional[str] = None,
        dataloader_num_workers: int = 0,
        max_prompt_length: int = 256,
        max_new_tokens: int = 32,   # borne de complétion côté GRPO
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_generations: int = 2,   # GRPO exige >= 2
        generation_batch_size: int = 2,  # multiple de num_generations
        max_steps: int = -1,        # -1 = pas de coupure anticipée
    ):
        os.makedirs(output_dir, exist_ok=True)

        # --- Sampling sain (évite TemperatureLogitsWarper avec 0.0) ---
        safe_temperature, do_sample = _sanitize_sampling_params(temperature)

        # --- Config de génération locale (certains flags peuvent être ignorés) ---
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if safe_temperature is not None:
            gen_kwargs["temperature"] = safe_temperature
        gen_cfg = GenerationConfig(**gen_kwargs)

        # --- Préparation GRPOConfig (avec alias pour compat TRL) ---
        grpoconfig_kwargs = dict(
            output_dir=output_dir,
            seed=seed,
            bf16=use_bf16,
            fp16=use_fp16,
            # vLLM (pris en compte uniquement si supporté par ta version)
            use_vllm=use_vllm,
            vllm_mode=vllm_mode,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            # Entraînement
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to=report_to,
            dataloader_num_workers=dataloader_num_workers,
            # Longueurs (nommage varie selon versions TRL)
            max_prompt_length=max_prompt_length,
            max_completion_length=max_new_tokens,      # nom le plus courant
            max_completions_length=max_new_tokens,     # alias potentiel
            # Génération côté GRPO (si supporté)
            top_p=top_p,
            num_generations=num_generations,
            generation_batch_size=generation_batch_size,
            # Contrôle des steps (HF Trainer attend un int; -1 = désactivé)
            max_steps=max_steps,
        )
        if safe_temperature is not None:
            grpoconfig_kwargs["temperature"] = safe_temperature

        # Filtrer selon signature réelle
        safe_grpoconfig_kwargs = _filter_kwargs_for_class(GRPOConfig, grpoconfig_kwargs)

        # Contraintes GRPO : ng >= 2 et batch multiple de ng
        if "num_generations" in safe_grpoconfig_kwargs:
            ng = int(safe_grpoconfig_kwargs["num_generations"])
            if ng < 2:
                safe_grpoconfig_kwargs["num_generations"] = 2
        if "num_generations" in safe_grpoconfig_kwargs and "generation_batch_size" in safe_grpoconfig_kwargs:
            ng = int(safe_grpoconfig_kwargs["num_generations"])
            gb = int(safe_grpoconfig_kwargs["generation_batch_size"])
            if gb % ng != 0:
                mult = max(1, (gb + ng - 1) // ng)
                safe_grpoconfig_kwargs["generation_batch_size"] = mult * ng

        # Instanciation et garde-fou max_steps
        self.grpo_config = GRPOConfig(**safe_grpoconfig_kwargs)
        # Assure un entier valide pour HF Trainer (évite None)
        try:
            if getattr(self.grpo_config, "max_steps", None) is None:
                setattr(self.grpo_config, "max_steps", -1)
            else:
                setattr(self.grpo_config, "max_steps", int(getattr(self.grpo_config, "max_steps")))
        except Exception:
            setattr(self.grpo_config, "max_steps", -1)

        # --- Préparation des kwargs pour GRPOTrainer (API varie selon versions TRL) ---
        trainer_kwargs = dict(
            model=model,
            args=self.grpo_config,
            train_dataset=train_dataset,
            # Variantes possibles : certaines versions prennent 'tokenizer' ou 'processing_class'
            tokenizer=tokenizer,
            processing_class=tokenizer,
            # Variantes du nom de la reward
            reward_funcs=math_accuracy_reward,
            reward_function=math_accuracy_reward,
            generation_config=gen_cfg,
            dataset_text_field="prompt",
        )
        safe_trainer_kwargs = _filter_kwargs_for_class(GRPOTrainer, trainer_kwargs)

        # Instanciation du trainer TRL
        self.trainer = GRPOTrainer(**safe_trainer_kwargs)

    def train(self):
        return self.trainer.train()

    def save_model(self):
        self.trainer.save_model()
