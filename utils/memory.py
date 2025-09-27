#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def report_memory(tag: str = "") -> None:
    """
    Utility function to report current GPU memory usage.
    This helps debug memory consumption at different phases of training
    (weights, activations, gradients, optimizer states, inference).

    Parameters
    ----------
    tag : str, optional
        A custom label to display in the log (e.g. "After forward").
    """
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return

    # Memory currently allocated by tensors (MB)
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    # Memory reserved by the caching allocator (MB)
    reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    # Maximum memory allocated so far during this run (MB)
    max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(
        f"[{tag}] "
        f"Allocated: {allocated_mb:.1f} MB | "
        f"Reserved: {reserved_mb:.1f} MB | "
        f"Max allocated: {max_allocated_mb:.1f} MB"
    )

