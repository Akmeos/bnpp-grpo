#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from config import Config
from model_loader import load_model_and_tokenizer


def main():
    cfg = Config()
    # Forcer un smoke test **CPU** pour éviter les bugs MPS avec PEFT/Granite sur Mac
    cfg.use_gpu = False
    prefer_mps = False   # <- clé : on évite MPS

    print("Chargement modèle/tokenizer…")
    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        use_gpu=cfg.use_gpu,
        prefer_mps=prefer_mps,
        lora_target_modules=cfg.lora_target_modules,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        freeze_experts=cfg.freeze_experts,
        train_router=cfg.train_router,
    )

    device = next(model.parameters()).device
    print(f"Modèle chargé sur : {device}")

    prompt = (
        "You are a careful math tutor. Solve step by step.\n\n"
        "Question: If I have 3 apples and buy 4 more, how many apples do I have?\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    # IMPORTANT : mettre les tenseurs sur le **même device** que le modèle
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    print("\n==== Sortie ====")
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
