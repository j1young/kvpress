#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RepoBench (v1.1) benchmark evaluation with Qwen3-8B + KV press.

Evaluates repository-level next-line code completion across three settings:
  - cross_file_first:  first usage of a cross-file module
  - cross_file_random: random usage of a cross-file module
  - in_file:           no cross-file dependency

The model receives cross-file context snippets followed by the in-file prefix,
then predicts the next line. Scored by Exact Match (EM) and Edit Similarity (ES).

Usage
-----
# Default: ExpectedAttentionPress, compression_ratio=0.0
python repobench_kvzip.py

# Multiple GPUs
python repobench_kvzip.py --n_gpus 4

# Different press / compression
python repobench_kvzip.py --press kvzip --compression_ratio 0.5

# Specific setting and level
python repobench_kvzip.py --setting cross_file_first --levels 2k 4k 8k

Dataset
-------
HuggingFace: tianyang/repobench_python_v1.1
  Splits: cross_file_first, cross_file_random, in_file

Columns:
  repo_name          : str  – GitHub repository name
  file_path          : str  – path of the file being completed
  context            : list – cross-file context snippets [{path, identifier, snippet}]
  import_statement   : str  – import statements of the target file
  cropped_code       : str  – code above the line to predict (the prefix)
  next_line          : str  – ground truth line to predict
  gold_snippet_index : int  – index of the most relevant context snippet
  level              : str  – prompt length bucket (2k, 4k, 8k, 12k, 16k, ...)

Scoring
-------
  Exact Match (EM): percentage of predictions matching ground truth (whitespace-tokenized)
  Edit Similarity (ES): average fuzzywuzzy.fuzz.ratio between prediction and ground truth

Design notes
------------
* No chat template — RepoBench uses raw code completion (FIM or plain prefix).
* For Qwen3 instruct models, we wrap the code prompt in a system+user message.
* Post-processing: extract first non-comment, non-empty line from output.
* Separator trick on user message to split context_ids / question_ids.
* Prefill:    model.model(context_ids, cache) inside with press(model):
* Generate:   manual greedy decode with position_ids
* Multi-GPU:  torch.multiprocessing.spawn with round-robin sharding.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import kvpress  # noqa: F401  – triggers patch_attention_functions() on import
from kvpress import ExpectedAttentionPress, FastKVzipPress, KVzapPress, KVzipPress

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Given the repository context and the current file, predict the exact next line of code. "
    "Output ONLY the single next line of code. Do not include explanations, comments, or multiple lines."
)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def construct_prompt(sample: dict) -> str:
    """
    Build the RepoBench code completion prompt from a dataset sample.

    Structure:
      # Cross-file context
      # Repo Name: {repo_name}
      # Path: {context[i].path}
      {context[i].snippet}
      ...

      # In-file context
      # Path: {file_path}
      {import_statement}
      {cropped_code}
    """
    parts = []

    # Cross-file context
    parts.append(f"# Repo Name: {sample['repo_name']}\n")
    for ctx in sample["context"]:
        path = ctx["path"] if isinstance(ctx, dict) else ctx.get("path", "")
        snippet = ctx["snippet"] if isinstance(ctx, dict) else ctx.get("snippet", "")
        parts.append(f"# Path: {path}\n{snippet}\n")

    # In-file context
    parts.append(f"# Path: {sample['file_path']}\n")
    if sample.get("import_statement"):
        parts.append(sample["import_statement"] + "\n")
    parts.append(sample["cropped_code"])

    prompt = "\n".join(parts)

    # Collapse runs of 4+ blank lines to 2 (same as original RepoBench)
    prompt = re.sub(r"\n{4,}", "\n\n", prompt)

    return prompt


def tokenize_prompt(
    tokenizer,
    code_prompt: str,
    max_context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize code_prompt into context_ids / question_ids using the
    separator trick (same pattern as bigcodebench_kvzip.py).

    context_ids  = chat template prefix + system msg + code prompt
    question_ids = chat template suffix (generation prompt tokens)
    """
    separator = "<<<SEP_REPOBENCH>>>"

    full_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": code_prompt + separator},
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    context_text, question_suffix = full_text.split(separator, maxsplit=1)

    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False)
    question_ids = tokenizer.encode(question_suffix, return_tensors="pt", add_special_tokens=False)

    if context_ids.shape[1] > max_context_length:
        logger.warning(
            f"Context truncated from {context_ids.shape[1]} to {max_context_length} tokens."
        )
        context_ids = context_ids[:, :max_context_length]

    return context_ids, question_ids


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(
    model,
    press,
    context_ids: torch.Tensor,
    question_ids: torch.Tensor,
    max_new_tokens: int,
    tokenizer,
) -> str:
    """
    Prefill with KV compression then greedily decode the answer.

    Steps
    -----
    1. model.model(context_ids, cache) inside with press(model):
    2. model(question_ids, cache, position_ids) → first output logit
    3. Greedy decode until EOS or newline or max_new_tokens
    """
    device = next(model.parameters()).device
    context_ids = context_ids.to(device)
    question_ids = question_ids.to(device)
    context_length = context_ids.shape[1]

    cache = DynamicCache()

    # ── 1. Prefill (press compression happens at context manager exit) ────────
    with press(model):
        model.model(
            input_ids=context_ids,
            past_key_values=cache,
        )

    # ── 2. Process question tokens ────────────────────────────────────────────
    position_ids = torch.arange(
        context_length,
        context_length + question_ids.shape[1],
        device=device,
    ).unsqueeze(0)

    outputs = model(
        input_ids=question_ids,
        past_key_values=cache,
        position_ids=position_ids,
        num_logits_to_keep=1,
    )

    # ── 3. Greedy decode ──────────────────────────────────────────────────────
    position_ids = position_ids[:, -1:] + 1
    generated_ids = [outputs.logits[0, -1].argmax()]

    stop_ids = model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]

    # Also stop on newline token
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    for step in range(max_new_tokens - 1):
        outputs = model(
            input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
            past_key_values=cache,
            position_ids=position_ids + step,
        )
        new_id = outputs.logits[0, -1].argmax()
        generated_ids.append(new_id)
        if new_id.item() in stop_ids:
            break
        # Stop at newline — we only need the next line
        if new_id.item() in newline_ids:
            break

    answer = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)
    return answer.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing and scoring
# ──────────────────────────────────────────────────────────────────────────────

def strip_thinking(raw_output: str) -> str:
    """Remove <think>...</think> block from Qwen3 thinking output."""
    return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()


def get_first_line_not_comment(text: str) -> str:
    """
    Extract the first non-empty, non-comment line from model output.
    This matches the RepoBench post-processing convention.
    """
    text = strip_thinking(text)
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    # Fallback: return the whole text stripped
    return text.strip()


def exact_match(pred: str, gt: str) -> float:
    """Whitespace-tokenized exact match (0.0 or 1.0)."""
    return 1.0 if pred.split() == gt.split() else 0.0


def edit_similarity(pred: str, gt: str) -> float:
    """Normalized edit similarity using fuzzywuzzy (0-100 scale)."""
    try:
        from fuzzywuzzy import fuzz
        return float(fuzz.ratio(pred, gt))
    except ImportError:
        # Fallback: simple ratio based on difflib
        import difflib
        return difflib.SequenceMatcher(None, pred, gt).ratio() * 100.0


def score_sample(pred: str, gt: str) -> dict:
    """Compute EM and ES for a single prediction."""
    em = exact_match(pred, gt)
    es = edit_similarity(pred, gt)
    return {"em": em, "es": es}


# ──────────────────────────────────────────────────────────────────────────────
# Per-GPU worker
# ──────────────────────────────────────────────────────────────────────────────

def worker(
    rank: int,
    world_size: int,
    samples: list,
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """
    Load a model copy on GPU `rank` and evaluate a shard of `samples`.
    Results are written to a per-rank JSON file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU{rank}] %(levelname)s %(message)s",
        force=True,
    )
    log = logging.getLogger(__name__)

    device = f"cuda:{rank}"
    log.info(f"Loading {args.model} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    press_map = {
        "kvzip": KVzipPress,
        "expected_attention": ExpectedAttentionPress,
        "kvzap": KVzapPress,
        "fastkvzip": FastKVzipPress,
    }
    press = press_map[args.press](compression_ratio=args.compression_ratio)

    # Round-robin shard
    shard = samples[rank::world_size]
    log.info(f"Processing {len(shard)} / {len(samples)} samples ...")

    results = []
    for idx, sample in enumerate(tqdm(shard, desc=f"GPU{rank}", position=rank, leave=True)):
        task_id = f"{sample.get('repo_name', 'unknown')}_{idx}"
        gt = sample["next_line"].strip()

        try:
            code_prompt = construct_prompt(sample)
            context_ids, question_ids = tokenize_prompt(
                tokenizer,
                code_prompt,
                args.max_context_length,
            )
            raw_output = run_inference(
                model=model,
                press=press,
                context_ids=context_ids,
                question_ids=question_ids,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            log.error(f"Inference failed for {task_id}: {exc}", exc_info=True)
            raw_output = ""

        pred = get_first_line_not_comment(raw_output)
        scores = score_sample(pred, gt)

        log.info(
            f"{task_id}: EM={scores['em']:.0f} ES={scores['es']:.1f}  "
            f"ctx_tokens={context_ids.shape[1]}  "
            f"level={sample.get('level', '?')}"
        )

        results.append({
            "task_id": task_id,
            "repo_name": sample.get("repo_name", ""),
            "level": sample.get("level", ""),
            "context_tokens": context_ids.shape[1],
            "prediction": pred,
            "ground_truth": gt,
            "raw_output": raw_output[:2000],
            "em": scores["em"],
            "es": scores["es"],
        })
        torch.cuda.empty_cache()

    shard_path = output_path.replace(".json", f"_shard{rank}.json")
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(results)} results → {shard_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_results(output_path: str, world_size: int) -> dict:
    """Merge per-GPU shards and compute EM / ES metrics (overall and per-level)."""
    all_results: list[dict] = []
    for rank in range(world_size):
        shard_path = output_path.replace(".json", f"_shard{rank}.json")
        with open(shard_path, encoding="utf-8") as f:
            all_results.extend(json.load(f))

    n_total = len(all_results)

    # Overall metrics
    avg_em = sum(r["em"] for r in all_results) / n_total * 100 if n_total > 0 else 0.0
    avg_es = sum(r["es"] for r in all_results) / n_total if n_total > 0 else 0.0

    # Per-level metrics
    levels: dict[str, list[dict]] = {}
    for r in all_results:
        lvl = r.get("level", "unknown")
        levels.setdefault(lvl, []).append(r)

    per_level = {}
    for lvl, items in sorted(levels.items()):
        n = len(items)
        per_level[lvl] = {
            "n": n,
            "em": round(sum(r["em"] for r in items) / n * 100, 2) if n > 0 else 0.0,
            "es": round(sum(r["es"] for r in items) / n, 2) if n > 0 else 0.0,
        }

    metrics = {
        "em": round(avg_em, 2),
        "es": round(avg_es, 2),
        "n_tasks": n_total,
        "per_level": per_level,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": all_results}, f, indent=2, ensure_ascii=False)

    # Clean up shards
    for rank in range(world_size):
        shard_path = output_path.replace(".json", f"_shard{rank}.json")
        if os.path.exists(shard_path):
            os.remove(shard_path)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RepoBench v1.1 evaluation with Qwen3-8B + KV press",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset", default="tianyang/repobench_python_v1.1",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--setting", default="cross_file_first",
        choices=["cross_file_first", "cross_file_random", "in_file"],
        help="RepoBench evaluation setting (dataset split)",
    )
    parser.add_argument(
        "--levels", nargs="+", default=["2k", "4k", "8k", "12k", "16k"],
        help="Prompt length levels to evaluate",
    )
    parser.add_argument(
        "--press", default="expected_attention",
        choices=["kvzip", "expected_attention", "kvzap", "fastkvzip"],
        help="KV press algorithm to use",
    )
    parser.add_argument(
        "--compression_ratio", type=float, default=0.0,
        help="Fraction of KV pairs to prune (0 = no compression)",
    )
    parser.add_argument(
        "--max_context_length", type=int, default=131072,
        help="Maximum context token length",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="Maximum tokens to generate (single line prediction)",
    )
    parser.add_argument(
        "--n_gpus", type=int, default=None,
        help="Number of GPUs to use. Defaults to all available.",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of dataset to evaluate",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--output_dir", default="./results/repobench",
        help="Directory for result JSON files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    # ── GPU count ─────────────────────────────────────────────────────────────
    n_available = torch.cuda.device_count()
    world_size = args.n_gpus if args.n_gpus is not None else n_available
    if world_size == 0:
        log.warning("No CUDA devices found; running on CPU with a single process.")
        world_size = 1
    log.info(f"Using {world_size} GPU(s) ({n_available} available)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info(f"Loading {args.dataset} (setting={args.setting}) ...")
    ds = load_dataset(args.dataset, split=args.setting)
    samples = ds.to_list()

    # Filter by level
    if args.levels:
        samples = [s for s in samples if s.get("level") in args.levels]
        log.info(f"Filtered to levels {args.levels}: {len(samples)} samples")

    if args.fraction < 1.0:
        import random
        random.seed(args.seed)
        k = max(1, int(len(samples) * args.fraction))
        samples = random.sample(samples, k)
        log.info(f"Sampled {k} examples (fraction={args.fraction})")

    log.info(f"Evaluating {len(samples)} tasks")

    # ── Output path ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "--")
    output_path = str(
        output_dir / f"{model_tag}_{args.setting}_{args.press}_cr{args.compression_ratio:.2f}.json"
    )
    log.info(f"Results → {output_path}")

    # ── Spawn workers ─────────────────────────────────────────────────────────
    if world_size == 1:
        worker(0, 1, samples, args, output_path)
    else:
        mp.spawn(
            worker,
            args=(world_size, samples, args, output_path),
            nprocs=world_size,
            join=True,
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    metrics = aggregate_results(output_path, world_size)

    log.info("\n=== Results ===")
    log.info(f"  Setting          : {args.setting}")
    log.info(f"  Press            : {args.press}")
    log.info(f"  Compression      : {args.compression_ratio}")
    log.info(f"  Exact Match (EM) : {metrics['em']:.2f}%")
    log.info(f"  Edit Similarity  : {metrics['es']:.2f}")
    log.info(f"  Total tasks      : {metrics['n_tasks']}")
    for lvl, m in metrics["per_level"].items():
        log.info(f"    {lvl:>5s}: EM={m['em']:.2f}%  ES={m['es']:.2f}  (n={m['n']})")
    log.info(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
