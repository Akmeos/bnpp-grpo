#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    # --- Exécution & ressources ---
    seed: int = 42
    use_gpu: bool = False                # Mac CPU = False ; Cloud GPU = True
    use_bf16: bool = False               # GPU Ampere+ -> True ; sinon False
    use_fp16: bool = False               # Alternative à bf16 si non supporté
    use_vllm: bool = False               # Mac CPU -> False ; GPU -> True
    vllm_mode: str = "colocate"          # "colocate" ou "server" (GPU only)
    vllm_gpu_memory_utilization: float = 0.5

    # --- Modèle HF (Granite 3.1 1B A400M Instruct) ---
    model_name: str = "ibm-granite/granite-3.1-1b-a400m-instruct"
    trust_remote_code: bool = True

    # --- LoRA / Couches ciblées ---
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    train_router: bool = True            # router entraîné
    freeze_experts: bool = True          # experts gelés

    # --- Données ---
    dataset_name: str = "gsm8k"          # HF dataset
    dataset_config: str = "main"         # "main" (standard)
    train_split: str = "train"
    eval_split: Optional[str] = None     # pas nécessaire pour smoke test
    max_train_samples: int = 32          # ↓ réduit pour CPU rapide
    max_eval_samples: int = 64

    # --- Génération (pendant GRPO) ---
    max_prompt_length: int = 256
    max_new_tokens: int = 32             # ↓ 32 pour éviter clipping sur CPU
    temperature: float = 0.0             # ↓ greedy pour aller vite en CPU
    top_p: float = 1.0

    # --- Entraînement GRPO ---
    output_dir: str = "outputs/grpo-granite"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    report_to: Optional[str] = None      # "tensorboard" sur GPU si tu veux

    # --- Debug / rapide ---
    debug_mode: bool = True              # True = plus rapide pour Mac
    dataloader_num_workers: int = 0      # Mac: 0 pour éviter soucis fork

    def to_dict(self):
        return asdict(self)
