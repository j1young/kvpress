#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Merge all rows of an MT-Eval subset into a single dialogue and save locally.

Each row's first turn is shared (duplicated) across rows, so only the first
row's first turn is kept; subsequent rows start from their second turn.

Usage
-----
python merge_mteval.py

python merge_mteval.py --subset expansion_multi --output merged_expansion_multi.json

The resulting JSON file can be passed to mteval_kvzip.py via --dataset:
  python mteval_kvzip.py --dataset merged_recollection_multi_cls.json
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all MT-Eval rows into a single dialogue JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="wckwan/MT-Eval",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--subset", default="recollection_multi_cls",
        help="Dataset config/subset name",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path. Defaults to merged_{subset}.json",
    )
    parser.add_argument(
        "--skip_first_turn", action="store_true", default=False,
        help="Skip the first (duplicated) turn of each row except the first row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    log.info(f"Loading {args.dataset} (subset={args.subset}, split={args.split}) ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    log.info(f"Total rows: {len(ds)}")

    merged_conv: list[dict] = []
    merged_id = f"merged_{args.subset}"

    for row_idx, sample in enumerate(ds):
        conv = sample["conv"]
        turns_to_add = conv if (row_idx == 0 or not args.skip_first_turn) else conv[1:]
        log.info(
            f"Row {row_idx} (id={sample['id']}): "
            f"{len(conv)} turns total, adding {len(turns_to_add)}"
        )
        merged_conv.extend(turns_to_add)

    n_inference = sum(1 for t in merged_conv if t["do_inference"])
    log.info(f"Merged dialogue: {len(merged_conv)} turns, {n_inference} inference turns")

    output_path = Path(args.output or f"merged_{args.subset}.json")
    result = {
        "id": merged_id,
        "conv": merged_conv,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"Saved to {output_path}")
    log.info(f"Use with: python mteval_kvzip.py --dataset {output_path}")


if __name__ == "__main__":
    main()
