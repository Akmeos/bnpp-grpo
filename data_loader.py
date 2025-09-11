#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset


def _build_prompt(question: str) -> str:
    # On force la forme de sortie compatible GSM8K pour faciliter la reward
    return (
        "You are a careful math tutor. Solve step by step. "
        "At the very end, write ONLY the final numeric answer on a new line as '#### <number>'.\n\n"
        f"Question: {question}\nAnswer:"
    )


def load_gsm8k_as_prompts(
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    limit: Optional[int] = None,
) -> Dataset:
    """
    Retourne un Dataset HF avec colonnes:
    - 'prompt': texte complet à donner au modèle
    - 'answer': gold (contient souvent '#### 42')
    """
    ds = load_dataset(dataset_name, dataset_config, split=split)

    def _map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        q = example.get("question", "").strip()
        a = example.get("answer", "").strip()
        return {"prompt": _build_prompt(q), "answer": a}

    keep = {"prompt", "answer"}
    ds = ds.map(_map_fn, remove_columns=[c for c in ds.column_names if c not in keep])
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def prepare_train_eval(
    dataset_name: str,
    dataset_config: str,
    train_split: str,
    eval_split: Optional[str],
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
) -> Dict[str, Dataset]:
    train_ds = load_gsm8k_as_prompts(dataset_name, dataset_config, train_split, max_train_samples)
    data = {"train": train_ds}
    if eval_split:
        eval_ds = load_gsm8k_as_prompts(dataset_name, dataset_config, eval_split, max_eval_samples)
        data["eval"] = eval_ds
    return data
