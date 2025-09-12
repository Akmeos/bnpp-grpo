#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

# --- Extraction & nettoyage des réponses ---
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

def clean_answer(ans: str) -> str:
    """
    Nettoie une réponse GSM8K pour ne garder que le nombre final.
    Exemple: 'The answer is #### 42' -> '42'
    """
    if not ans:
        return ans
    m = RE_HASH.search(ans)
    if m:
        return m.group(1)
    return ans.strip()


def get_dataloaders(
    tokenizer,
    num_samples: int = 64,
    batch_size: int = 1,
    shuffle: bool = True,
    max_input_tokens: int = 768,
):
    """
    Charge GSM8K avec un prompting CoT (Chain of Thought).
    Prépare prompts et réponses attendues.
    """

    dataset = load_dataset("gsm8k", "main")

    # Sélection d'un sous-échantillon pour l'entraînement rapide
    train_data = dataset["train"].shuffle(seed=42).select(range(min(num_samples, len(dataset["train"]))))

    prompts, answers = [], []
    for ex in train_data:
        q = ex["question"]
        a = clean_answer(ex["answer"])

        # Prompt CoT
        prompt = (
            "You are a helpful math tutor. Solve the problem step by step. "
            "Show concise reasoning, then provide ONLY the final numeric answer "
            "on a new line starting with '#### '.\n\n"
            f"Q: {q}\nA: Let's reason step by step."
        )

        prompts.append(prompt)
        answers.append(a)

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
