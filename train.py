#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for GRPO (Grouped Reinforcement Policy Optimization) on GSM8K.
Handles argument parsing, configuration setup, and training pipeline initialization.
"""

import argparse
import os
import random
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from data_loader import get_dataloaders
from model_loader import load_model_and_tokenizer
from grpo_trainer import GRPOConfig, GRPOTrainerWrapper


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Global seed set to {seed}")


def main():
    # Argument parsing for training configuration
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

    # --- Seed Setup ---
    set_seed(args.seed)

    # --- Logging Setup ---
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard logging enabled in {args.log_dir}")

    # --- Metadata Storage ---
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        f.write(
            f'{{"seed": {args.seed}, "samples": {args.samples}, '
            f'"max_steps": {args.max_steps}, "new_tokens": {args.new_tokens}}}\n'
        )
    print(f"Run metadata saved to {args.output_dir}/run_meta.json")

    # --- Model and Tokenizer Loading ---
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # --- Dataset Preparation ---
    print("Preparing GSM8K dataset...")
    train_loader = get_dataloaders(
        tokenizer,
        num_samples=args.samples,
        batch_size=1,           # Batch size 1 for RL training
        shuffle=True,
        max_input_tokens=768,   # Reasonable input length for math problems
    )
    print(f"Dataset loaded with {len(train_loader)} samples")

    # --- GRPO Configuration ---
    cfg = GRPOConfig(
        learning_rate=5e-5,     # Stable learning rate for T4 GPU
        weight_decay=0.01,      # L2 regularization
        warmup_steps=50,        # Learning rate warmup
        max_grad_norm=1.0,      # Gradient clipping
        clip_range=0.2,         # PPO clip range
        kl_coef=0.05,           # KL divergence coefficient
        entropy_coef=0.001,     # Low entropy bonus for exploration
        normalize_rewards=True, # Reward normalization
        save_steps=50,          # Checkpoint frequency
        do_sample=True,         # Sampling for diversity
        temperature=args.temperature,
    )

    # --- Trainer Initialization ---
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
    )

    # --- Training Execution ---
    print("Starting training...")
    print("=" * 60)
    trainer.train(train_loader)
    print("=" * 60)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()