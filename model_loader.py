#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_model_and_tokenizer(model_name: str = "ibm-granite/granite-3.1-1b-a400m-instruct"):
    """
    Charge Granite 3.1 1b a400m (MoE) + applique LoRA sur q/k/v/o_proj.
    On gèle tous les poids, sauf :
      - LoRA (q_proj, k_proj, v_proj, o_proj)
      - Router MoE (router.layer.weight) -> trainable "plein"
    Les experts restent gelés.
    """
    print(f"Chargement du modèle {model_name} ...")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Évite erreurs de padding à la génération
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Modèle ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # --- Tout geler par défaut ---
    for p in model.parameters():
        p.requires_grad = False

    # --- Dégeler router MoE (plein entraînement du routeur) ---
    for name, p in model.named_parameters():
        # Granite MoE: "block_sparse_moe.router.layer.weight"
        if "block_sparse_moe.router.layer.weight" in name:
            p.requires_grad = True

    # --- LoRA sur les proj d'attention ---
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention seulement
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # --- Debug: affichage des paramètres entraînables ---
    model.print_trainable_parameters()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Paramètres] Total : {total_params:,} | Trainables : {trainable_params:,} | Figés : {total_params - trainable_params:,}")

    # Exemple de noms trainables (utile pour valider router + lora)
    shown = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  + {n}")
            shown += 1
            if shown >= 20:
                break

    return model, tokenizer
