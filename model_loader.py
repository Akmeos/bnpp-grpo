#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import re

MODEL_NAME = "ibm-granite/granite-3.1-1b-a400m-instruct"

def set_trainable_attention_and_router(model):
    """
    Gèle tous les paramètres sauf :
    - Projections Q/K/V/O de l'attention
    - Router/gating des experts
    """
    n_trainable, n_frozen = 0, 0

    # Tout geler
    for name, param in model.named_parameters():
        param.requires_grad = False
        n_frozen += param.numel()

    # Définir les patterns
    patterns = [
        r"\b(attn|attention)\.(q_proj|k_proj|v_proj|o_proj)\b",
        r"\b(router|gating|gate)\b",
    ]

    def is_target(name):
        return any(re.search(pat, name) for pat in patterns)

    # Dé-geler attention + router
    for name, param in model.named_parameters():
        if is_target(name):
            param.requires_grad = True
            n_trainable += param.numel()

    print(f"[Paramètres] Trainables : {n_trainable:,} | Figés : {n_frozen - n_trainable:,}")
    print("Exemples de paramètres actifs :")
    for name, p in list((n, p) for n, p in model.named_parameters() if p.requires_grad)[:20]:
        print("  +", name)

    return model

def load_model_with_lora():
    """
    Charge Granite 1.3B (MoE), applique LoRA sur attention et gèle les experts.
    """
    print(f"Chargement du modèle {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Appliquer LoRA sur l'attention
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention uniquement
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # Forcer seulement attention + router comme trainables
    model = set_trainable_attention_and_router(model)

    model.print_trainable_parameters()
    return model

def load_tokenizer():
    """
    Charge le tokenizer de Granite.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
