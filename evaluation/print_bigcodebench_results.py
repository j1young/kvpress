#!/usr/bin/env python3
"""Print BigCodeBench results as TSV to stdout."""

import json
import sys
from pathlib import Path

results_dir = Path("./results/bigcodebench")

print("press\tcompression_ratio\tthinking\tpass_at_1\tn_passed\tn_tasks")

for path in sorted(results_dir.glob("*.json")):
    # Parse filename: ..._{press}_cr{ratio}_{thinking}.json
    stem = path.stem
    parts = stem.rsplit("_", 3)  # [..., press, crX.XX, thinking]
    if len(parts) < 4:
        continue
    thinking = parts[-1]  # "think" or "nothink"
    cr_str = parts[-2]    # "cr0.50"
    press = parts[-3]     # "kvzip", "expected_attention", etc.

    # Handle press names with underscores (e.g., "expected_attention", "fastkvzip")
    # Re-parse: find crX.XX pattern and work from there
    cr_idx = stem.rfind("_cr")
    thinking_idx = stem.rfind("_")
    thinking = stem[thinking_idx + 1:]
    cr_str = stem[cr_idx + 1:thinking_idx]

    # Find press name: between split version and _cr
    # Format: ..._{split}_{press}_cr{ratio}_{thinking}
    prefix = stem[:cr_idx]
    # split is "v0.1.4", find it
    split_idx = prefix.find("_v0.")
    if split_idx == -1:
        continue
    # After split, next underscore starts press name
    after_split = prefix[split_idx + 1:]  # "v0.1.4_expected_attention"
    first_us = after_split.find("_")
    press = after_split[first_us + 1:]

    compression_ratio = cr_str.replace("cr", "")

    with open(path) as f:
        data = json.load(f)
    m = data["metrics"]

    print(f"{press}\t{compression_ratio}\t{thinking}\t{m['pass_at_1']}\t{m['n_passed']}\t{m['n_tasks']}")
