#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Extraction numérique robuste
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
RE_ANY  = re.compile(r"-?\d+(?:\.\d+)?")

def clean_answer(ans: str) -> str:
    """
    Extraction plus robuste des nombres de la réponse
    """
    if not ans:
        return ""
    
    # Priorité au nombre après ####
    m = RE_HASH.search(ans)
    if m:
        return m.group(1)
    
    # Sinon chercher le dernier nombre dans le texte
    nums = RE_ANY.findall(ans)
    if nums:
        return nums[-1]
    
    return ""


# Prompt concis avec format imposé
INSTR = (
    "You are a helpful math tutor. Solve the problem.\n"
    "Output ONLY the final numeric answer in the exact format: '#### number'\n"
    "Example: If the answer is 42, output: #### 42\n"
    "Do not output any other text, explanations, or formatting.\n"
)

def get_dataloaders(
    tokenizer,
    num_samples: int = 64,
    batch_size: int = 1,
    shuffle: bool = True,
    max_input_tokens: int = 768,
):
    """
    Charge GSM8K (train), formate des paires (prompt, answer_clean).
    On force la génération en terminant le prompt par '#### ' pour obtenir directement un nombre.
    """
    ds = load_dataset("gsm8k", "main")
    train = ds["train"].shuffle(seed=42).select(range(min(num_samples, len(ds["train"]))))

    prompts, answers = [], []
    for ex in train:
        q = ex["question"]
        a_clean = clean_answer(ex["answer"])
        # Prompt + préfixe '#### ' pour forcer le format
        prompt = f"{INSTR}\nQ: {q}\nA:\n#### "
        prompts.append(prompt)
        answers.append(a_clean)

    class RLDS(Dataset):
        def __init__(self, prompts, answers):
            self.prompts = prompts
            self.answers = answers

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            return {"prompts": self.prompts[idx], "answers": self.answers[idx]}

    return DataLoader(RLDS(prompts, answers), batch_size=batch_size, shuffle=shuffle)
