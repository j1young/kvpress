#!/usr/bin/env python3
"""Collect MT-Eval JSON results into a TSV summary."""

import json
import sys
from pathlib import Path


def main():
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./results/mteval/merged_refinement_multi")

    rows = []
    for p in sorted(result_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        cfg = data["config"]
        rows.append({
            "model": cfg["model"],
            "subset": cfg.get("subset") or data.get("dialogue_id", ""),
            "press": cfg["press"],
            "compression_ratio": cfg["compression_ratio"],
            "avg_score": round(data["avg_score"], 2),
        })

    cols = ["model", "subset", "press", "compression_ratio", "avg_score"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
