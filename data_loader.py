#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader for GSM8K dataset with reinforcement learning formatting.
Handles data loading, answer extraction, and prompt formatting for math reasoning tasks.
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Robust numeric extraction patterns
RE_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")  # Extract number after ####
RE_ANY = re.compile(r"-?\d+(?:\.\d+)?")            # Extract any number in text


def clean_answer(answer_text: str) -> str:
    """
    Extract clean numeric answer from GSM8K response text.
    
    Priority:
    1. Number following '####' pattern (official format)
    2. Last number in the text (fallback)
    3. Empty string if no number found
    
    Args:
        answer_text: Raw answer string from GSM8K dataset
        
    Returns:
        Clean numeric string or empty string
    """
    if not answer_text:
        return ""
    
    # Priority 1: Extract number after #### (official format)
    hash_match = RE_HASH.search(answer_text)
    if hash_match:
        return hash_match.group(1)
    
    # Priority 2: Extract last number in text (fallback)
    numbers = RE_ANY.findall(answer_text)
    if numbers:
        return numbers[-1]
    
    return ""


# Instruction prompt with strict output formatting
INSTRUCTION_PROMPT = (
    "Answer with ONLY the number in this format: #### number\n"
    "No text, no explanation, just #### followed by the number.\n"
)


def get_dataloaders(
    tokenizer,
    num_samples: int = 64,
    batch_size: int = 1,
    shuffle: bool = True,
    max_input_tokens: int = 768,
):
    """
    Load and format GSM8K dataset for reinforcement learning training.
    
    Creates prompt-answer pairs with strict output formatting to guide the model
    towards generating only numeric answers in the required format.
    
    Args:
        tokenizer: Tokenizer for text processing
        num_samples: Number of training samples to use
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        max_input_tokens: Maximum input tokens for truncation
        
    Returns:
        DataLoader: Formatted dataset for RL training
    """
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"].shuffle(seed=42).select(
        range(min(num_samples, len(dataset["train"])))
    )

    prompts, answers = [], []
    
    # Format each example with strict prompt structure
    for example in train_data:
        question = example["question"]
        clean_ans = clean_answer(example["answer"])
        
        # Construct prompt with forced output format
        prompt = f"Q: {question}\nA: #### "
        prompts.append(prompt)
        answers.append(clean_ans)

    # Custom Dataset class for RL training
    class RLMathDataset(Dataset):
        def __init__(self, prompts, answers):
            self.prompts = prompts
            self.answers = answers

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            return {
                "prompts": self.prompts[idx],
                "answers": self.answers[idx]
            }

    # Create DataLoader with specified parameters
    return DataLoader(
        RLMathDataset(prompts, answers),
        batch_size=batch_size,
        shuffle=shuffle
    )


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-instruct")
    dataloader = get_dataloaders(tokenizer, num_samples=10)
    
    print("DataLoader created successfully!")
    print(f"Number of batches: {len(dataloader)}")
    
    # Show first example
    first_batch = next(iter(dataloader))
    print(f"Prompt: {first_batch['prompts'][0][:100]}...")
    print(f"Answer: {first_batch['answers'][0]}")