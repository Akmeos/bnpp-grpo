#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def _pick_device(use_gpu: bool, prefer_mps: bool) -> str:
    """
    Ordre de choix :
    - si use_gpu et CUDA dispo -> 'cuda'
    - sinon si prefer_mps True ET MPS dispo -> 'mps'
    - sinon -> 'cpu'
    """
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_map_from_device(device: str):
    if device in ("cpu", "mps"):
        return {"": device}
    return "auto"  # CUDA


def _freeze_experts_but_router(model, train_router: bool):
    """
    Gèle les 'experts' et (dé)gèle le routeur selon train_router.
    """
    for name, param in model.named_parameters():
        lname = name.lower()
        if "experts" in lname:
            param.requires_grad = False
        elif ("router" in lname) or ("gate" in lname):
            param.requires_grad = bool(train_router)
        # le reste sera géré par LoRA


def load_model_and_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    use_gpu: bool = False,
    prefer_mps: bool = False,          # <- NOUVEAU : False par défaut pour éviter les bugs MPS
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    freeze_experts: bool = True,
    train_router: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    device = _pick_device(use_gpu=use_gpu, prefer_mps=prefer_mps)
    device_map = _device_map_from_device(device)

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,          # API récente
            **extra_kwargs,
        )
    except TypeError:
        warnings.warn(
            "Transformers ne supporte pas 'dtype', fallback vers 'torch_dtype'. "
            "Mettez à jour Transformers si possible."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,    # fallback
            **extra_kwargs,
        )

    if freeze_experts:
        _freeze_experts_but_router(model, train_router=train_router)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Sur CPU/MPS, on force explicitement le device
    if device in ("cpu", "mps"):
        model.to(device)

    return model, tokenizer
