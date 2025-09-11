#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Dict
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve step by step. "
    "At the very end, write ONLY the final numeric answer on a new line as '#### <number>' "
    "and then STOP."
)

def build_prompt(question: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

def prepare_train_eval(
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    train_split: str = "train",
    eval_split: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Dict[str, Dataset]:
    # --- train ---
    ds_train = load_dataset(dataset_name, dataset_config, split=train_split)
    ds_train = ds_train.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"]),
            "answer": ex["answer"],  # texte complet GSM8K (contient '#### <nombre>')
        },
        remove_columns=ds_train.column_names,  # ne conserver que ce que l'on retourne
    )
    if max_train_samples is not None:
        ds_train = ds_train.select(range(min(max_train_samples, len(ds_train))))

    out: Dict[str, Dataset] = {"train": ds_train}

    # --- eval (optionnel) ---
    if eval_split:
        ds_eval = load_dataset(dataset_name, dataset_config, split=eval_split)
        ds_eval = ds_eval.map(
            lambda ex: {
                "prompt": build_prompt(ex["question"]),
                "answer": ex["answer"],
            },
            remove_columns=ds_eval.column_names,
        )
        if max_eval_samples is not None:
            ds_eval = ds_eval.select(range(min(max_eval_samples, len(ds_eval))))
        out["eval"] = ds_eval

    return out
