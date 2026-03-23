#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RULER benchmark evaluation with Qwen3-8B + KVzipPress.

N GPUs split the dataset and run data-parallel evaluation.
Each (context, question) sample gets fresh KV compression via KVzipPress.

Usage
-----
# Single GPU, context length 4096 (default)
python ruler_kvzip.py

# 4 GPUs
python ruler_kvzip.py --n_gpus 4

# Different context length
python ruler_kvzip.py --context_length 8192 --n_gpus 4

Dataset
-------
HuggingFace: simonjegou/ruler
  data_dir / config_name = context length string: "4096", "8192", "16384"

Columns: context, question, answer_prefix, answer (list[str]), task, max_new_tokens

Tasks (13 total):
  NIAH  : niah_single_1/2/3, niah_multikey_1/2/3, niah_multivalue, niah_multiquery
  QA    : qa_1, qa_2
  VT    : vt
  CWE   : cwe
  FWE   : fwe

Scoring
-------
  QA tasks   → string_match_part : any ref in prediction
  All others → string_match_all  : fraction of refs in prediction
  Score range: 0–100

Design notes
------------
* context_ids  = chat-template prefix + context text  (separator trick, same as pipeline.py)
* question_ids = question + chat-template suffix + answer_prefix
* Prefill:    model.model(context_ids, cache)  inside  with press(model):
              → KVzip performs chunked reconstruction and sets masked_key_indices
* Generate:   manual greedy decode with position_ids, matching pipeline.py pattern
* Multi-GPU:  torch.multiprocessing.spawn; each process owns one GPU + one data shard
              → results written to per-GPU JSON shards, aggregated by rank-0
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import kvpress  # noqa: F401  – triggers patch_attention_functions() on import
from kvpress import KVzipPress

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_context_and_question(
    tokenizer,
    context: str,
    question: str,
    answer_prefix: str,
    max_context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build and tokenize context_ids / question_ids, following pipeline.py's
    preprocess pattern exactly.

    The separator trick embeds a unique marker at the end of the user content
    so we can split 'context_text' from 'question_suffix' (the chat-template
    closing + assistant-opening tokens) after apply_chat_template.

    Returns
    -------
    context_ids  : (1, ctx_len)  – context text up to the separator
    question_ids : (1, q_len)    – question + question_suffix + answer_prefix
    """
    separator = "<<<SEP_RULER>>>"

    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + separator}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    context_text, question_suffix = full_text.split(separator, maxsplit=1)
    # question_suffix = e.g. "<|im_end|>\n<|im_start|>assistant\n" for Qwen3

    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False)
    question_text = question + question_suffix + answer_prefix
    question_ids = tokenizer.encode(question_text, return_tensors="pt", add_special_tokens=False)

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
    press: KVzipPress,
    context_ids: torch.Tensor,
    question_ids: torch.Tensor,
    max_new_tokens: int,
    tokenizer,
) -> str:
    """
    Prefill with KVzip compression then greedily decode the answer.

    Mirrors pipeline.py _forward + generate_answer without the Pipeline wrapper.

    Steps
    -----
    1. model.model(context_ids, cache)  inside  with press(model):
       → KVzip runs chunked reconstruction, sets masked_key_indices
    2. model(question_ids, cache, position_ids)  → first output logit
    3. Greedy decode until EOS or max_new_tokens
    """
    device = next(model.parameters()).device
    context_ids = context_ids.to(device)
    question_ids = question_ids.to(device)
    context_length = context_ids.shape[1]

    cache = DynamicCache()

    # ── 1. Prefill (KVzip compression happens at context manager exit) ────────
    # Use model.model (not model) to bypass the LM head, matching pipeline.py.
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

    answer = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)
    return answer.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Scoring  (mirrors calculate_metrics.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

_NP_PATTERN = re.compile(r"[\x00-\x1f]")


def _clean(text: str) -> str:
    return _NP_PATTERN.sub("", text.strip()).strip()


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    """Any reference present in prediction (for QA tasks). Returns 0–100."""
    score = (
        sum(max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) for pred, ref in zip(preds, refs))
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    """Fraction of references present in prediction (for non-QA tasks). Returns 0–100."""
    score = (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
            for pred, ref in zip(preds, refs)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


def score_sample(prediction: str, refs: list[str], task: str) -> float:
    """Return per-sample score in [0, 1] (not scaled to 100)."""
    pred = _clean(prediction)
    task_category = task.split("_")[0]
    if task_category == "qa":
        return max(1.0 if r.lower() in pred.lower() else 0.0 for r in refs)
    else:
        return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


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

    Results are written to a per-rank JSON file; the main process aggregates
    them after all workers finish.
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
        trust_remote_code=True,
    )
    model.eval()

    press = KVzipPress(compression_ratio=args.compression_ratio)

    # Each GPU processes every world_size-th sample (round-robin shard)
    shard = samples[rank::world_size]
    log.info(f"Processing {len(shard)} / {len(samples)} samples ...")

    results = []
    for sample in tqdm(shard, desc=f"GPU{rank}", position=rank, leave=True):
        context = sample["context"]
        question = sample["question"]
        answer_prefix = sample.get("answer_prefix", "")
        task = sample["task"]
        max_new_tokens = sample.get("max_new_tokens", 128)

        refs = sample["answer"]
        if isinstance(refs, str):
            refs = [refs]

        try:
            context_ids, question_ids = tokenize_context_and_question(
                tokenizer,
                context,
                question,
                answer_prefix,
                args.max_context_length,
            )
            prediction = run_inference(
                model=model,
                press=press,
                context_ids=context_ids,
                question_ids=question_ids,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            log.error(f"Sample failed (task={task}): {exc}", exc_info=True)
            prediction = ""

        per_sample_score = score_sample(prediction, refs, task)
        results.append({
            "task": task,
            "context_tokens": context_ids.shape[1],
            "question": question,
            "answer": refs,
            "answer_prefix": answer_prefix,
            "predicted_answer": prediction,
            "score": per_sample_score,
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
    """
    Merge per-GPU shards and compute per-task string_match scores,
    mirroring calculate_metrics.py exactly.
    """
    all_results: list[dict] = []
    for rank in range(world_size):
        shard_path = output_path.replace(".json", f"_shard{rank}.json")
        with open(shard_path, encoding="utf-8") as f:
            all_results.extend(json.load(f))

    # Group by task
    by_task: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_task[r["task"]].append(r)

    task_scores: dict[str, dict] = {}
    for task, rows in sorted(by_task.items()):
        preds = [_clean(r["predicted_answer"]) for r in rows]
        refs = [r["answer"] for r in rows]
        task_category = task.split("_")[0]
        metric_fn = string_match_part if task_category == "qa" else string_match_all
        task_scores[task] = {"string_match": metric_fn(preds, refs), "n": len(rows)}

    # Overall = mean of per-task scores
    overall = round(
        sum(v["string_match"] for v in task_scores.values()) / len(task_scores), 2
    ) if task_scores else 0.0

    metrics = {
        "context_length": None,  # filled by caller
        "overall_string_match": overall,
        "by_task": task_scores,
        "n_samples": len(all_results),
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
        description="RULER benchmark evaluation with Qwen3-8B + KVzipPress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset", default="simonjegou/ruler",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--context_length", type=int, default=4096,
        help="RULER context length variant to evaluate. "
             "Used as data_dir when loading the dataset. "
             "Available: 4096, 8192, 16384.",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--compression_ratio", type=float, default=0.5,
        help="Fraction of KV pairs to prune (0 = no compression)",
    )
    parser.add_argument(
        "--max_context_length", type=int, default=None,
        help="Hard cap on context token length (default: context_length arg × 1.1 for safety). "
             "Contexts exceeding this are truncated.",
    )
    parser.add_argument(
        "--n_gpus", type=int, default=None,
        help="Number of GPUs to use. Defaults to all available CUDA devices.",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of dataset to evaluate (useful for quick checks)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--output_dir", default="./results/ruler",
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

    # Default max_context_length = context_length × 1.1 (small overhead buffer)
    if args.max_context_length is None:
        args.max_context_length = int(args.context_length * 1.1)

    # ── GPU count ─────────────────────────────────────────────────────────────
    n_available = torch.cuda.device_count()
    world_size = args.n_gpus if args.n_gpus is not None else n_available
    if world_size == 0:
        log.warning("No CUDA devices found; running on CPU with a single process.")
        world_size = 1
    log.info(f"Using {world_size} GPU(s) ({n_available} available)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_dir = str(args.context_length)
    log.info(f"Loading {args.dataset} (data_dir={data_dir}, split={args.split}) ...")
    ds = load_dataset(args.dataset, data_dir=data_dir, split=args.split)
    samples = ds.to_list()

    if args.fraction < 1.0:
        import random
        random.seed(args.seed)
        k = max(1, int(len(samples) * args.fraction))
        samples = random.sample(samples, k)
        log.info(f"Sampled {k} / {len(ds)} examples (fraction={args.fraction})")

    task_counts = defaultdict(int)
    for s in samples:
        task_counts[s["task"]] += 1
    log.info(f"Evaluating on {len(samples)} samples across {len(task_counts)} tasks:")
    for task, cnt in sorted(task_counts.items()):
        log.info(f"  {task}: {cnt}")

    # ── Output path ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "--")
    output_path = str(
        output_dir / f"{model_tag}_ctx{args.context_length}_kvzip_cr{args.compression_ratio:.2f}.json"
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
    metrics["context_length"] = args.context_length

    # Re-write with context_length filled in
    with open(output_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["metrics"]["context_length"] = args.context_length
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.truncate()

    log.info("\n=== Results ===")
    log.info(f"  Context length : {args.context_length}")
    log.info(f"  Overall        : {metrics['overall_string_match']:.2f}")
    log.info(f"  N samples      : {metrics['n_samples']}")
    log.info("  Per-task breakdown:")
    for task, m in metrics["by_task"].items():
        log.info(f"    {task:30s}  string_match={m['string_match']:6.2f}  n={m['n']}")
    log.info(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
