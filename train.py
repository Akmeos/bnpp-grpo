#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import time
import threading
from typing import Optional

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


# ------- NVML sampler (optionnel, pour pic VRAM global) -------
class GPUMemorySampler:
    """
    Échantillonne la VRAM utilisée (global driver) pendant l'entraînement.
    Utile pour capturer l'usage vLLM/Autres lib hors tracking PyTorch.
    Requiert: pip install pynvml (sur la machine GPU).
    """
    def __init__(self, device_index: int = 0, interval_sec: float = 0.5):
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ok = False
        try:
            import pynvml  # type: ignore
            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._ok = True
        except Exception:
            self._ok = False

    def start(self):
        if not self._ok:
            return

        def loop():
            while not self._stop.is_set():
                try:
                    info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                    # info.used en bytes
                    if info.used > self.peak_bytes:
                        self.peak_bytes = info.used
                except Exception:
                    pass
                time.sleep(self.interval_sec)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._ok:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._ok


# ------- Profils d’exécution -------
def build_cfg_from_profile(profile: str, args) -> Config:
    cfg = Config()

    # overrides via CLI
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
        # CPU / Mac — run court
        cfg.use_gpu = False
        cfg.max_train_samples = cfg.max_train_samples or 32
        cfg.max_new_tokens = cfg.max_new_tokens or 32
        if args.temperature is None:
            cfg.temperature = 0.0
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 1
        cfg.num_train_epochs = 1
        cfg.use_vllm = False
        cfg.dataloader_num_workers = 0

    elif profile == "gpu":
        # Cloud — 1x GPU <= 16 Go
        cfg.use_gpu = True
        cfg.use_bf16 = True          # ou fp16 si bf16 non dispo
        cfg.use_vllm = not args.no_vllm
        cfg.vllm_mode = "colocate"
        cfg.vllm_gpu_memory_utilization = 0.45
        cfg.max_new_tokens = cfg.max_new_tokens or 64
        if args.temperature is None:
            cfg.temperature = 0.7
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 8
        cfg.num_train_epochs = 1
    else:
        raise ValueError(f"Profil inconnu: {profile}")

    return cfg


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
                        help="Mesurer la VRAM max (PyTorch + NVML si dispo).")
    parser.add_argument("--nvml-interval", type=float, default=0.5,
                        help="Période d'échantillonnage NVML en secondes.")
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
        prefer_mps=False,  # éviter MPS avec PEFT
        lora_target_modules=cfg.lora_target_modules,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        freeze_experts=cfg.freeze_experts,
        train_router=cfg.train_router,
    )

    # GRPO (2 générations mini)
    num_generations = 2
    generation_batch_size = 2

    # --- Tracking mémoire ---
    if args.track-mem if False else False:
        # (éviter parse error; voir bloc juste après)
        pass
    # correct implementation:
    nvml_sampler = None
    if args.track_mem and torch.cuda.is_available():
        # Reset des stats PyTorch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # NVML sampler (optionnel)
        nvml_sampler = GPUMemorySampler(device_index=0, interval_sec=args.nvml_interval)
        if nvml_sampler.available:
            nvml_sampler.start()
        else:
            nvml_sampler = None

    # --- Entraînement + timing ---
    start = time.perf_counter()

    trainer = GRPOTrainerWrapper(
        model=model,
        tokenizer=tokenizer,
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
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        max_steps=args.max_steps,
    )

    trainer.train()
    trainer.save_model()

    elapsed = time.perf_counter() - start

    # Arrêt sampler NVML si actif
    if nvml_sampler is not None:
        nvml_sampler.stop()

    # --- Reporting métriques runtime/mémoire ---
    print("\n================ Runtime & Mémoire ================")
    print(f"Temps d'entraînement (mur): {elapsed:.2f} s")
    if torch.cuda.is_available():
        # Mémoire PyTorch (process courant)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"PyTorch CUDA peak — allocated: {peak_alloc:.2f} GB | reserved: {peak_reserved:.2f} GB")
        # Mémoire globale observée via NVML (si dispo)
        if nvml_sampler is not None:
            peak_nvml = nvml_sampler.peak_bytes / (1024 ** 3)
            print(f"NVML peak observed (global): {peak_nvml:.2f} GB")
            if peak_nvml > 16.0:
                print("⚠️  Alerte: pic VRAM > 16 Go (condition dépassée).")
        else:
            print("(NVML indisponible — pic VRAM global non échantillonné)")
    else:
        print("CUDA non disponible — pas de VRAM à reporter (CPU/MPS).")
    print("===================================================\n")

    print(f"Entraînement GRPO terminé. Modèle sauvegardé dans: {cfg.output_dir}")


if __name__ == "__main__":
    main()
