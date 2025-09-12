#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import load_gsm8k_dataset
from model_loader import load_model_and_tokenizer
from grpo_trainer import GRPOTrainerWrapper, GRPOConfig


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
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Seed
    set_seed(args.seed)
    print(f"✅ Seed global fixé à {args.seed}")

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Chargement modèle/tokenizer
    model, tokenizer = load_model_and_tokenizer("ibm-granite/granite-3.1-1b-a400m-instruct")
    tokenizer.pad_token = tokenizer.eos_token  # fix pad token

    # Dataset
    train_data = load_gsm8k_dataset(split="train[:{}]".format(args.samples))
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    # Trainer
    config = GRPOConfig(temperature=args.temperature)
    trainer = GRPOTrainerWrapper(
        model,
        tokenizer,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        temperature=args.temperature,
        new_tokens=args.new_tokens,
        no_vllm=args.no_vllm,
        writer=writer,
        config=config,
    )

    # Train
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
