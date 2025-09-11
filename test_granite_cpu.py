#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"

def main():
    print("Chargement modèle/tokenizer (CPU)…")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float32, device_map=None
    ).to("cpu")

    prompt = (
        "You are a careful math tutor. Solve step by step.\n\n"
        "Question: If I have 3 apples and buy 4 more, how many apples do I have?\n"
        "Answer:"
    )
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    print("\n==== Sortie ====\n")
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
