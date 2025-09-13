#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration dataclass for GRPO training with IBM Granite MoE model.
Centralizes all training parameters for easy experimentation and reproducibility.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class Config:
    """
    Comprehensive configuration for GRPO training with Granite MoE model.
    
    Groups parameters into logical sections:
    - Execution & Resources
    - Model Configuration
    - LoRA Adaptation
    - Dataset Settings
    - Generation Parameters
    - Training Hyperparameters
    - Reward Shaping
    """
    
    # --- Execution & Resources ---
    seed: int = 42
    use_gpu: bool = False                  # Mac CPU = False ; Cloud/Kaggle GPU = True
    use_bf16: bool = False                 # T4 doesn't support bf16
    use_fp16: bool = True                  # fp16 works on T4
    use_vllm: bool = False                 # Kaggle: False ; Cloud: True if desired
    vllm_mode: str = "colocate"            # "colocate" or "server" (GPU only)
    vllm_gpu_memory_utilization: float = 0.40

    # --- HF Model (Granite 3.1 1B A400M Instruct) ---
    model_name: str = "ibm-granite/granite-3.1-1b-a400m-instruct"
    trust_remote_code: bool = True

    # --- LoRA / Target Modules ---
    lora_r: int = 8                        # LoRA rank
    lora_alpha: int = 16                   # LoRA alpha scaling
    lora_dropout: float = 0.05             # LoRA dropout rate
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    train_router: bool = True              # Train router parameters
    freeze_experts: bool = True            # Freeze expert layers

    # --- Dataset Settings ---
    dataset_name: str = "gsm8k"            # GSM8K math reasoning dataset
    dataset_config: str = "main"           # Dataset configuration
    train_split: str = "train"             # Training split
    eval_split: Optional[str] = None       # Evaluation split (optional)
    max_train_samples: int = 256           # Maximum training samples
    max_eval_samples: int = 64             # Maximum evaluation samples

    # --- Generation / GRPO Parameters ---
    max_prompt_length: int = 256           # Maximum prompt tokens
    max_new_tokens: int = 64               # Maximum new tokens to generate
    temperature: float = 0.8               # Sampling temperature for diversity
    top_p: float = 1.0                     # Nucleus sampling parameter

    # --- GRPO Training ---
    output_dir: str = "outputs/grpo-granite"  # Output directory
    num_train_epochs: int = 1              # Number of training epochs
    per_device_train_batch_size: int = 1   # Batch size per device
    gradient_accumulation_steps: int = 8   # Gradient accumulation steps
    learning_rate: float = 2e-4            # Learning rate
    lr_scheduler_type: str = "cosine"      # Learning rate scheduler
    warmup_ratio: float = 0.05             # Warmup ratio
    weight_decay: float = 0.0              # Weight decay
    logging_steps: int = 10                # Logging frequency
    save_steps: int = 500                  # Checkpoint saving frequency
    save_total_limit: int = 2              # Maximum checkpoints to keep
    report_to: Optional[str] = None        # Reporting destination
    dataloader_num_workers: int = 0        # DataLoader workers (0 for Kaggle)

    # --- Reward Shaping ---
    reward_format_bonus: float = 0.10      # Bonus for correct '####' format
    reward_missing_penalty: float = 0.02   # Penalty for missing '####' format

    debug_mode: bool = False               # Debug mode flag

    def to_dict(self):
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)


# Example usage
if __name__ == "__main__":
    config = Config()
    print("Configuration loaded with default values:")
    for key, value in config.to_dict().items():
        print(f"{key:30}: {value}")