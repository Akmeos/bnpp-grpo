#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_dataloaders
from grpo_trainer import GRPOTrainerWrapper, GRPOConfig
from model_loader import load_model_with_lora, load_tokenizer


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-granite")
    parser.add_argument("--log_dir", type=str, default="outputs/grpo-granite/logs")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no_vllm", dest="no_vllm", action="store_true")
    parser.add_argument("--no-vllm", dest="no_vllm", action="store_true")  # alias
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Seed & dirs
    set_seed(args.seed)
    print(f"✅ Seed global fixé à {args.seed}")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard + meta
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_scalar("meta/seed", args.seed, 0)
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump({"seed": args.seed}, f, indent=2)
    print(f"ℹ️ Meta run sauvegardée dans {os.path.join(args.output_dir, 'run_meta.json')}")

    # Charger modèle + tokenizer
    model = load_model_with_lora()
    tokenizer = load_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # ✅ évite les soucis de sampling/padding

    # Option de reprise
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        print(f"🔄 Reprise depuis {args.resume_from_checkpoint}")
        try:
            model = model.from_pretrained(args.resume_from_checkpoint)
            tokenizer = tokenizer.from_pretrained(args.resume_from_checkpoint)
        except Exception as e:
            print("⚠️ Impossible de charger le checkpoint, on continue fresh:", e)

    total, trainable, frozen = count_parameters(model)
    print(f"[Paramètres] Total : {total:,} | Trainables : {trainable:,} | Figés : {frozen:,}")

    # DataLoader RL
    train_loader = get_dataloaders(
        tokenizer=tokenizer,
        num_samples=args.samples,
        batch_size=1,          # ✅ PPO = batch=1 conseillé
        shuffle=True,
        max_input_tokens=768,
    )

    # Trainer GRPO
    cfg = GRPOConfig(
        temperature=args.temperature,
        do_sample=False,     # ✅ Greedy par défaut (stabilité Kaggle/T4)
    )
    trainer = GRPOTrainerWrapper(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        temperature=args.temperature,
        new_tokens=args.new_tokens,
        no_vllm=args.no_vllm,
        writer=writer,
        config=cfg,
    )

    # Entraînement
    trainer.train(train_loader)

    # Sauvegarde finale
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    writer.close()
    print(f"✅ Entraînement terminé. Modèle et tokenizer sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
