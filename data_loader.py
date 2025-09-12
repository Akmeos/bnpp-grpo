#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

# --- Patterns numériques ---
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
RE_ANY  = re.compile(r"-?\d+(?:\.\d+)?")

def clean_answer(ans: str) -> str:
    """
    Renvoie la version 'label' la plus exploitable côté RL:
    - Priorité au nombre après '####'
    - Sinon, si un nombre existe dans le texte, renvoie ce nombre (dernier)
    - Sinon, renvoie ans.strip() (fallback)
    """
    if not ans:
        return ""
    m = RE_HASH.search(ans)
    if m:
        return m.group(1)
    nums = list(RE_ANY.finditer(ans))
    if nums:
        return nums[-1].group(0)
    return ans.strip()


# Petit few-shot CoT pour stabiliser la baseline
FEW_SHOTS = """You are a helpful math tutor. Solve the problem step by step. Show concise reasoning, then provide ONLY the final numeric answer on a new line starting with '#### '.

Q: If a notebook costs 3 dollars and a pen costs 2 dollars, how much do 4 notebooks and 3 pens cost in total?
A: Let's reason step by step.
4 notebooks cost 4 × 3 = 12 dollars.
3 pens cost 3 × 2 = 6 dollars.
Total = 12 + 6 = 18 dollars.
#### 18

Q: A box has 24 apples. If 3 friends share them equally, how many apples does each friend get?
A: Let's reason step by step.
We divide 24 apples by 3 friends: 24 ÷ 3 = 8.
Each friend gets 8 apples.
#### 8
"""


def get_dataloaders(
    tokenizer,
    num_samples: int = 64,
    batch_size: int = 1,
    shuffle: bool = True,
    max_input_tokens: int = 768,
):
    """
    Charge GSM8K (split 'train'), applique un prompt CoT avec few-shots,
    et retourne un DataLoader de paires (prompt, answer_clean).
    """

    ds = load_dataset("gsm8k", "main")

    # Sous-échantillon pour run rapide
    train_data = ds["train"].shuffle(seed=42).select(range(min(num_samples, len(ds["train"]))))

    prompts, answers = [], []
    for ex in train_data:
        q = ex["question"]
        a_clean = clean_answer(ex["answer"])

        # Prompt CoT : few-shot + question courante
        prompt = (
            FEW_SHOTS.strip()
            + "\n\n"
            + f"Q: {q}\nA: Let's reason step by step."
        )

        prompts.append(prompt)
        answers.append(a_clean)

    class RLDS(torch.utils.data.Dataset):
        def __init__(self, prompts, answers):
            self.prompts = prompts
            self.answers = answers

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            return {"prompts": self.prompts[idx], "answers": self.answers[idx]}

    dataset = RLDS(prompts, answers)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
