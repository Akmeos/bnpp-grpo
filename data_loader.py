##!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Prépare GSM8K pour GRPO : prompting CoT + extraction de la réponse finale.
# On renvoie un DataLoader qui fournit des dicts {"prompts": [...], "answers": [...], "answers_raw": [...]}

from typing import List, Dict, Any, Optional
import re
import random

from datasets import load_dataset
from torch.utils.data import DataLoader

# ---------- Few-shots & Prompting CoT ----------

FEW_SHOTS: List[dict] = [
    {
        "q": "If a notebook costs 3 dollars and a pen costs 2 dollars, how much do 4 notebooks and 3 pens cost in total?",
        "r": (
            "Let's reason step by step.\n"
            "4 notebooks cost 4 × 3 = 12 dollars.\n"
            "3 pens cost 3 × 2 = 6 dollars.\n"
            "Total = 12 + 6 = 18 dollars.\n"
            "#### 18"
        ),
    },
    {
        "q": "A box has 24 apples. If 3 friends share them equally, how many apples does each friend get?",
        "r": (
            "Let's reason step by step.\n"
            "We divide 24 apples by 3 friends: 24 ÷ 3 = 8.\n"
            "Each friend gets 8 apples.\n"
            "#### 8"
        ),
    },
]

INSTRUCTION = (
    "You are a helpful math tutor. Solve the problem step by step. "
    "Show concise reasoning, then provide ONLY the final numeric answer on a new line "
    "starting with '#### '."
)

def build_cot_prompt(question: str, few_shots: List[dict] = FEW_SHOTS) -> str:
    """
    Construit un prompt de raisonnement (CoT) court et déterministe.
    La réponse attendue doit finir par une ligne '#### <nombre>'.
    """
    parts: List[str] = [INSTRUCTION, ""]
    for ex in few_shots:
        parts.append("Q: " + ex["q"])
        parts.append("A: " + ex["r"])
        parts.append("")  # ligne vide de séparation
    parts.append("Q: " + question.strip())
    parts.append("A:")
    return "\n".join(parts)

# ---------- Extraction de la réponse finale pour la reward ----------

_FINAL_ANS_RE = re.compile(r"####\s*([^\n\r]+)")

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extrait la réponse finale au format GSM8K (après '#### ').
    Retourne la chaîne telle quelle (peut contenir fractions, décimales, signes).
    """
    m = _FINAL_ANS_RE.search(text)
    if not m:
        return None
    ans = m.group(1).strip()

    # Nettoyages légers usuels GSM8K (enlève ponctuation finale, espaces)
    ans = ans.rstrip(".").strip()
    # Supprime éventuelles balises latex simples
    ans = ans.replace("\\boxed{", "").replace("}", "").strip()
    return ans if ans else None

# ---------- Prétraitement GSM8K ----------

def preprocess_examples(
    examples: Dict[str, List[str]],
    tokenizer,
    max_input_tokens: int = 768,
) -> Dict[str, List[str]]:
    """
    Construit le prompt CoT + extrait la réponse finale de GSM8K.
    - 'prompt' : instruction + few-shots + question
    - 'answer' : réponse finale nettoyée (après ####)
    - 'answer_raw' : champ 'answer' original GSM8K
    """
    prompts: List[str] = []
    answers: List[str] = []
    answers_raw: List[str] = []

    qs = examples["question"]
    ars = examples["answer"]

    for q, a_raw in zip(qs, ars):
        prompt = build_cot_prompt(q)
        # Optionnel: tronquer côté input si trop long
        toks = tokenizer(prompt, add_special_tokens=False)
        if len(toks["input_ids"]) > max_input_tokens:
            prompt = tokenizer.decode(
                toks["input_ids"][:max_input_tokens], skip_special_tokens=True
            )

        a_final = extract_final_answer(a_raw) or ""  # fallback chaîne vide si non trouvé
        prompts.append(prompt)
        answers.append(a_final)
        answers_raw.append(a_raw)

    return {"prompt": prompts, "answer": answers, "answer_raw": answers_raw}

# ---------- DataLoader pour GRPO ----------

def _collate_prompts(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Collate pour RL: on garde les strings brutes (pas de tokenization ici),
    c'est le trainer qui gère la génération.
    """
    return {
        "prompts": [b["prompt"] for b in batch],
        "answers": [b["answer"] for b in batch],           # réponse finale nettoyée
        "answers_raw": [b["answer_raw"] for b in batch],   # pour debug éventuel
    }

def get_dataloaders(
    tokenizer,
    num_samples: int = 256,
    seed: int = 42,
    batch_size: int = 1,
    shuffle: bool = True,
    max_input_tokens: int = 768,
):
    """
    Charge GSM8K (split train), applique le prompting CoT et renvoie un DataLoader.
    - num_samples : nombre d'exemples utilisés pour l'entraînement
    - batch_size : 1 conseillé pour GRPO/ppo-style avec génération
    """
    # Fixe la seed Python (datasets gère déjà sa propre seed)
    random.seed(seed)

    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"].shuffle(seed=seed)
    if num_samples is not None and num_samples > 0:
        num_samples = min(num_samples, len(train_ds))
        train_ds = train_ds.select(range(num_samples))

    processed = train_ds.map(
        lambda batch: preprocess_examples(batch, tokenizer, max_input_tokens=max_input_tokens),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Preparing GSM8K with CoT prompting",
    )

    # Sanity check (imprime un exemple de prompt & réponse attendue)
    if len(processed) > 0:
        print("=== Example CoT prompt (truncated) ===")
        print(processed[0]["prompt"][:800])
        print("=== Expected final answer (clean) ===")
        print(processed[0]["answer"])

    loader = DataLoader(
        processed,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_prompts,
    )
    return loader
