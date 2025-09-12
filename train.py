##!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import torch
import json
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_dataloaders
from grpo_trainer import GRPOTrainerWrapper
from model_loader import load_model_with_lora, load_tokenizer


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def set_seed(seed: int):
    """Fixe toutes les seeds pour reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print("⚠️ Deterministic note:", e)
    print(f"✅ Seed global fixé à {seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-granite")
    parser.add_argument("--log_dir", type=str, default="outputs/grpo-granite/logs")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # --- Seed & reproductibilité ---
    set_seed(args.seed)

    # --- TensorBoard writer ---
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_scalar("meta/seed", args.seed, 0)

    # --- Sauvegarder meta run ---
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"seed": args.seed}, f, indent=2)
    print(f"ℹ️ Meta run sauvegardée dans {meta_path}")

    # --- Charger modèle + tokenizer ---
    model = load_model_with_lora()
    tokenizer = load_tokenizer()

    # --- Reprise depuis un checkpoint ---
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            print(f"🔄 Reprise depuis {args.resume_from_checkpoint}")
            model = model.from_pretrained(args.resume_from_checkpoint)
            tokenizer = tokenizer.from_pretrained(args.resume_from_checkpoint)
        else:
            print(f"⚠️ Checkpoint {args.resume_from_checkpoint} introuvable, démarrage fresh.")

    total, trainable, frozen = count_parameters(model)
    print(f"[Paramètres] Total : {total:,} | Trainables : {trainable:,} | Figés : {frozen:,}")

    # --- Data ---
    train_loader = get_dataloaders(tokenizer, args.samples)

    # --- Trainer ---
    trainer = GRPOTrainerWrapper(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        temperature=args.temperature,
        new_tokens=args.new_tokens,
        no_vllm=args.no_vllm,
        writer=writer
    )

    # --- Entraînement ---
    trainer.train(train_loader)

    # Sauvegarde finale
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    writer.close()
    print(f"✅ Entraînement terminé. Modèle et tokenizer sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
