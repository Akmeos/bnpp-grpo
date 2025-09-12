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
    Label 'propre' pour le RL :
    - Priorité au nombre après '####'
    - Sinon dernier nombre trouvé dans le texte
    - Sinon ans.strip()
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

INSTR = (
    "You are a helpful math tutor. Solve the problem.\n"
    "Output ONLY the final numeric answer prefixed by '#### ' and nothing else.\n"
)

def get_dataloaders(
    tokenizer,
    num_samples: int = 64,
    batch_size: int = 1,
    shuffle: bool = True,
    max_input_tokens: int = 768,
):
    """
    Charge GSM8K (train), prépare des paires (prompt, answer_clean).
    On force la génération en commençant par '#### ' pour garantir un nombre.
    """
    ds = load_dataset("gsm8k", "main")
    train = ds["train"].shuffle(seed=42).select(range(min(num_samples, len(ds["train"]))))

    prompts, answers = [], []
    for ex in train:
        q = ex["question"]
        a_clean = clean_answer(ex["answer"])
        # ❗ Prompt court + contrainte de format et préfixe '#### '
        prompt = f"{INSTR}\nQ: {q}\nA:\n#### "
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

    return DataLoader(RLDS(prompts, answers), batch_size=batch_size, shuffle=shuffle)
