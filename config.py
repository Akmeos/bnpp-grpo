#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

@dataclass
class Config:
    # --- Exécution & ressources ---
    seed: int = 42
    use_gpu: bool = False                  # Mac CPU = False ; Cloud/Kaggle GPU = True
    use_bf16: bool = False                 # T4 ne supporte pas bf16
    use_fp16: bool = True                  # fp16 OK sur T4
    use_vllm: bool = False                 # Kaggle: False ; Cloud: True si souhaité
    vllm_mode: str = "colocate"            # "colocate" ou "server" (GPU only)
    vllm_gpu_memory_utilization: float = 0.40

    # --- Modèle HF (Granite 3.1 1B A400M Instruct) ---
    model_name: str = "ibm-granite/granite-3.1-1b-a400m-instruct"
    trust_remote_code: bool = True

    # --- LoRA / Cibles ---
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    train_router: bool = True              # router entraîné
    freeze_experts: bool = True            # experts gelés

    # --- Données ---
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    eval_split: Optional[str] = None
    max_train_samples: int = 256
    max_eval_samples: int = 64

    # --- Génération / GRPO ---
    max_prompt_length: int = 256
    max_new_tokens: int = 64
    temperature: float = 0.8               # diversité intra-groupe
    top_p: float = 1.0

    # --- Entraînement GRPO ---
    output_dir: str = "outputs/grpo-granite"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    report_to: Optional[str] = None
    dataloader_num_workers: int = 0        # Kaggle: 0

    # --- Reward shaping léger ---
    reward_format_bonus: float = 0.10      # + si '####' présent
    reward_missing_penalty: float = 0.02   # - si '####' absent

    debug_mode: bool = False

    def to_dict(self):
        return asdict(self)
