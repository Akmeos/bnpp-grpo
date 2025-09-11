#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Iterable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def _set_trainable_params(
    model: torch.nn.Module,
    train_router: bool = True,
    freeze_experts: bool = True,
    router_keywords: Iterable[str] = ("router", "gate"),
    experts_keyword: str = "experts",
):
    for name, p in model.named_parameters():
        # par défaut, on gèle tout (les LoRA injectées seront entraînables)
        p.requires_grad = False

    if freeze_experts:
        for name, p in model.named_parameters():
            if experts_keyword in name:
                p.requires_grad = False  # explicit, pour clarté

    if train_router:
        for name, p in model.named_parameters():
            if any(k in name for k in router_keywords):
                p.requires_grad = True

def load_model_and_tokenizer(
    model_name: str,
    trust_remote_code: bool,
    use_gpu: bool,
    prefer_mps: bool = False,
    lora_target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    freeze_experts: bool = True,
    train_router: bool = True,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    # dtype auto -> fp16/bf16 si possible, sinon float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )

    # (optionnel) checkpointing pour gratter de la VRAM sur GPU
    try:
        base_model.gradient_checkpointing_enable()
    except Exception:
        pass

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    # Router entraîné + experts gelés
    _set_trainable_params(model, train_router=train_router, freeze_experts=freeze_experts)

    # déplacer sur CPU explicite si pas de CUDA
    if device == "cpu":
        model.to("cpu")

    return model, tok
