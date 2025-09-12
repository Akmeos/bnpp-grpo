#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    """Fixe toutes les seeds pour la reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed global fixé à {seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-granite")
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32)   # court car '#### ' imposé
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # --- Seed ---
    set_seed(args.seed)

    # --- Logging ---
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # --- Meta run ---
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        f.write(
            f'{{"seed": {args.seed}, "samples": {args.samples}, '
            f'"max_steps": {args.max_steps}, "new_tokens": {args.new_tokens}}}\n'
        )
    print(f"ℹ️ Meta run sauvegardée dans {args.output_dir}/run_meta.json")

    # --- Charger modèle + tokenizer ---
    model, tokenizer = load_model_and_tokenizer()

    # --- Charger dataset GSM8K ---
    train_loader = get_dataloaders(
        tokenizer,
        num_samples=args.samples,
        batch_size=1,
        shuffle=True,
        max_input_tokens=768,
    )

    # --- Config GRPO ---
    cfg = GRPOConfig(
        learning_rate=5e-5,     # stable sur T4
        weight_decay=0.01,
        warmup_steps=50,
        max_grad_norm=1.0,
        clip_range=0.2,
        kl_coef=0.05,
        entropy_coef=0.001,     # entropie faible
        normalize_rewards=True,
        save_steps=50,
        do_sample=False,        # greedy => évite erreurs multinomial CUDA
        temperature=args.temperature,
    )

    # --- Trainer ---
    trainer = GRPOTrainerWrapper(
        model,
        tokenizer,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        temperature=args.temperature,
        new_tokens=args.new_tokens,
        writer=writer,
        config=cfg,
    )

    # --- Entraînement ---
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
