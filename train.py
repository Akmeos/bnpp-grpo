#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for GRPO (Grouped Reinforcement Policy Optimization) on GSM8K.
Handles argument parsing, configuration setup, and training pipeline initialization.

Improvements:
 - Explicit attention_mask handling (fix empty generations)
 - Custom reward function with regex check on last number
"""

import argparse
import os
import random
import re
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from data_loader import get_dataloaders
from model_loader import load_model_and_tokenizer
from grpo_trainer import GRPOConfig, GRPOTrainerWrapper
from utils.memory import report_memory


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Global seed set to {seed}")


def reward_function(output: str, gold: int) -> float:
    """
    Custom reward function.
    Reward = 1 if the last number in the model output matches the gold label.
    Reward = 0 otherwise.
    """
    numbers = re.findall(r"\d+", output)
    if not numbers:
        return 0.0
    try:
        return 1.0 if int(numbers[-1]) == gold else 0.0
    except ValueError:
        return 0.0


def main():
    # -----------------------
    # Argument parsing
    # -----------------------
    parser = argparse.ArgumentParser(description="GRPO Training for GSM8K Math Reasoning")

    parser.add_argument("--output_dir", type=str, default="outputs/grpo-granite",
                        help="Directory to save model checkpoints and final model")
    parser.add_argument("--log_dir", type=str, default="outputs/logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--samples", type=int, default=64,
                        help="Number of training samples from GSM8K")
    parser.add_argument("--new-tokens", type=int, default=32,
                        help="Maximum new tokens to generate (short due to '#### ' prefix)")
    parser.add_argument("--max-steps", type=int, default=120,
                        help="Maximum training steps")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # -----------------------
    # Seed setup
    # -----------------------
    set_seed(args.seed)

    # -----------------------
    # Logging setup
    # -----------------------
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard logging enabled in {args.log_dir}")

    # -----------------------
    # Metadata storage
    # -----------------------
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        f.write(
            f'{{"seed": {args.seed}, "samples": {args.samples}, '
            f'"max_steps": {args.max_steps}, "new_tokens": {args.new_tokens}}}\n'
        )
    print(f"Run metadata saved to {args.output_dir}/run_meta.json")

    # -----------------------
    # Model and tokenizer
    # -----------------------
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    report_memory("After model load (weights)")

    # -----------------------
    # Dataset preparation
    # -----------------------
    print("Preparing GSM8K dataset...")
    train_loader = get_dataloaders(
        tokenizer,
        num_samples=args.samples,
        batch_size=1,            # RL training = batch size 1
        shuffle=True,
        max_input_tokens=768,
    )
    print(f"Dataset loaded with {len(train_loader)} samples")

    # -----------------------
    # Dry-run inference test
    # -----------------------
    print("Running a dry-run inference to measure KV cache usage...")
    sample_batch = next(iter(train_loader))

    # ✅ Add attention_mask explicitly
    inputs = tokenizer(
        sample_batch["prompts"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    _ = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=32,
    )
    report_memory("During inference (KV cache + activations)")

    # -----------------------
    # GRPO configuration
    # -----------------------
    cfg = GRPOConfig(
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        clip_range=0.2,
        kl_coef=0.05,
        entropy_coef=0.001,
        normalize_rewards=True,
        save_steps=50,
        do_sample=True,
        temperature=args.temperature,
    )

    # -----------------------
    # Trainer init
    # -----------------------
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainerWrapper(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        temperature=args.temperature,
        new_tokens=args.new_tokens,
        writer=writer,
        config=cfg,
        reward_fn=reward_function,  # ✅ custom reward function
    )

    # -----------------------
    # Training execution
    # -----------------------
    print("Starting training...")
    print("=" * 60)
    trainer.train(train_loader)
    print("=" * 60)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
