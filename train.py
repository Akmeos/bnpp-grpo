#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import time

import numpy as np
import torch

from config import Config
from data_loader import prepare_train_eval
from model_loader import load_model_and_tokenizer
from grpo_trainer import GRPOTrainerWrapper


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cfg_from_profile(profile: str, args) -> Config:
    cfg = Config()

    # Overrides via CLI
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.samples is not None:
        cfg.max_train_samples = args.samples
    if args.new_tokens is not None:
        cfg.max_new_tokens = args.new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.log_tb:
        cfg.report_to = "tensorboard"

    if profile == "local":
        cfg.use_gpu = False
        cfg.max_train_samples = cfg.max_train_samples or 32
        cfg.max_new_tokens = cfg.max_new_tokens or 32
        cfg.temperature = 0.0 if args.temperature is None else cfg.temperature
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 1
        cfg.num_train_epochs = 1
        cfg.use_vllm = False
        cfg.dataloader_num_workers = 0

    elif profile == "gpu":
        cfg.use_gpu = True
        cfg.use_bf16 = False   # T4: False
        cfg.use_fp16 = True
        cfg.use_vllm = not args.no_vllm
        cfg.vllm_mode = "colocate"
        cfg.vllm_gpu_memory_utilization = 0.40
        cfg.max_new_tokens = cfg.max_new_tokens or 64
        cfg.temperature = 0.8 if args.temperature is None else cfg.temperature
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 8
        cfg.num_train_epochs = 1
        cfg.dataloader_num_workers = 0
    else:
        raise ValueError(f"Profil inconnu: {profile}")

    return cfg


def format_gb(b: int) -> str:
    return f"{b / (1024 ** 3):.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="GRPO + LoRA on Granite (GSM8K)")
    parser.add_argument("--profile", choices=["local", "gpu"], default="local")
    parser.add_argument("--samples", type=int, default=None,
                        help="Nombre max d'exemples d'entraînement (None=full).")
    parser.add_argument("--new-tokens", type=int, default=None,
                        help="max_new_tokens (longueur de complétion).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Température de génération (0.0 = greedy).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dossier de sortie/checkpoints.")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Couper l'entraînement après N steps (-1 = désactivé).")
    parser.add_argument("--log-tb", action="store_true",
                        help="Reporter les métriques vers TensorBoard.")
    parser.add_argument("--no-vllm", action="store_true",
                        help="(GPU) Désactiver vLLM.")
    parser.add_argument("--track-mem", action="store_true",
                        help="Afficher la VRAM max PyTorch (CUDA uniquement).")
    parser.add_argument("--num-generations", type=int, default=2,
                        help="Nombre de générations par prompt (GRPO ≥ 2).")
    parser.add_argument("--gen-batch-size", type=int, default=2,
                        help="Taille batch génération (multiple de --num-generations).")
    parser.add_argument("--disable-tqdm", action="store_true",
                        help="Masquer la barre de progression.")
    args = parser.parse_args()

    cfg = build_cfg_from_profile(args.profile, args)
    set_seed(cfg.seed)

    # Données
    dataset_dict = prepare_train_eval(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        train_split=cfg.train_split,
        eval_split=cfg.eval_split,
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )
    train_ds = dataset_dict["train"]

    # Modèle + Tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        use_gpu=cfg.use_gpu,
        prefer_mps=False,  # éviter MPS
        lora_target_modules=cfg.lora_target_modules,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        freeze_experts=cfg.freeze_experts,
        train_router=cfg.train_router,
    )

    # Mémoire PyTorch
    if args.track_mem and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    trainer = GRPOTrainerWrapper(
        model=model,
        train_dataset=train_ds,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        use_bf16=cfg.use_bf16,
        use_fp16=cfg.use_fp16,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=1 if args.profile == "local" else cfg.logging_steps,
        save_steps=999999 if args.profile == "local" else cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        dataloader_num_workers=cfg.dataloader_num_workers,
        max_prompt_length=cfg.max_prompt_length,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        num_generations=args.num_generations,
        generation_batch_size=args.gen_batch_size,
        max_steps=args.max_steps,
        disable_tqdm=args.disable_tqdm,
        reward_format_bonus=cfg.reward_format_bonus,
        reward_missing_penalty=cfg.reward_missing_penalty,
    )

    trainer.train()
    trainer.save_model()

    elapsed = time.perf_counter() - start

    # Reporting
    print("\n================ Runtime & Mémoire (PyTorch) ================")
    print(f"Temps d'entraînement (mur): {elapsed:.2f} s")
    if args.track_mem:
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            print(f"CUDA peak allocated : {format_gb(peak_alloc)}")
            print(f"CUDA peak reserved  : {format_gb(peak_reserved)}")
            if peak_reserved > 16 * 1024 ** 3:
                print("⚠️  Alerte: pic VRAM réservée > 16 Go (condition dépassée).")
        else:
            print("CUDA non disponible — aucune VRAM à reporter (CPU/MPS).")
    print("=============================================================\n")

    print(f"Entraînement GRPO terminé. Modèle sauvegardé dans: {cfg.output_dir}")


if __name__ == "__main__":
    main()
