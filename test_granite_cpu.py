#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model loading and basic inference test script for IBM Granite MoE model.
Verifies model can be loaded and performs basic text generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model identifier from Hugging Face Hub
MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"

def main():
    """Load model and tokenizer, then perform basic inference test."""
    print("Loading model and tokenizer (CPU)...")
    
    # Load tokenizer with trust_remote_code for custom tokenization
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model with float32 precision for CPU compatibility
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        trust_remote_code=True, 
        torch_dtype=torch.float32, 
        device_map=None
    ).to("cpu")

    # Test prompt for mathematical reasoning
    prompt = (
        "You are a careful math tutor. Solve step by step.\n\n"
        "Question: If I have 3 apples and buy 4 more, how many apples do I have?\n"
        "Answer:"
    )
    
    # Tokenize input prompt
    inputs = tok(prompt, return_tensors="pt")
    
    # Generate response with deterministic sampling
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    
    # Decode and print generated output
    print("\n==== Generated Output ====\n")
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()