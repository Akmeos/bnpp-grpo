#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model loader for IBM Granite MoE 1.3B model with LoRA adaptation.
Handles model loading, tokenizer setup, and parameter freezing strategy.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_model_and_tokenizer(model_name: str = "ibm-granite/granite-3.1-1b-a400m-instruct"):
    """
    Load IBM Granite 3.1 1B MoE model and apply LoRA adaptation.
    
    Strategy:
    - Freeze all parameters by default
    - Train router layers fully (MoE routing decisions)
    - Apply LoRA only to attention projections (q_proj, k_proj, v_proj, o_proj)
    - Keep expert layers frozen for efficiency
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        model: PEFT model with LoRA adaptation
        tokenizer: Tokenizer with appropriate settings
    """
    print(f"Loading model {model_name}...")

    # --- Tokenizer Configuration ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure padding token is set to avoid generation errors
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for batch processing

    # --- Model Loading ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # FP16 for memory efficiency
        device_map="auto",          # Automatic device placement
    )

    # --- Parameter Freezing Strategy ---
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # --- Unfreeze MoE Router ---
    # Train router fully for better expert selection
    for name, param in model.named_parameters():
        if "block_sparse_moe.router.layer.weight" in name:
            param.requires_grad = True
            print(f"Unfrozen router parameter: {name}")

    # --- LoRA Configuration ---
    lora_config = LoraConfig(
        r=8,                           # Rank of LoRA adaptation
        lora_alpha=16,                 # Scaling factor
        lora_dropout=0.05,             # Dropout for LoRA layers
        target_modules=[               # Attention projections only
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        task_type="CAUSAL_LM",         # Causal language modeling
    )
    model = get_peft_model(model, lora_config)

    # --- Parameter Statistics ---
    model.print_trainable_parameters()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"[Parameters] Total: {total_params:,} | "
          f"Trainable: {trainable_params:,} | "
          f"Frozen: {frozen_params:,}")

    # --- Display Trainable Parameters (for verification) ---
    print("Top trainable parameters:")
    shown = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  + {name}")
            shown += 1
            if shown >= 20:  # Show first 20 trainable parameters
                break

    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    model, tokenizer = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully!")