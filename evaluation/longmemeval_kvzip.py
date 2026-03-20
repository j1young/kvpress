#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LongMemEval benchmark evaluation with Qwen3-8B + KVzipPress.

N GPUs split the dataset and run data-parallel evaluation.
Each sample gets fresh KV compression via KVzipPress (no cache reuse across samples).

Usage
-----
# Single GPU
python longmemeval_kvzip.py

# 4 GPUs
python longmemeval_kvzip.py --n_gpus 4

# Custom settings
python longmemeval_kvzip.py --compression_ratio 0.5 --n_gpus 2 --subset oracle

Dataset
-------
HuggingFace: xiaowu0162/longmemeval
  Subsets: oracle (gold sessions only), s500 (500-session haystack), s200, ...

Key fields per sample:
  question_id        : unique identifier
  question           : question to answer
  expected_answer    : str or list[str] ground truth(s)
  haystack_sessions  : list[list[{role, content, (date)}]] – all past conversation sessions
  question_type      : e.g. "single-session-user", "multi-session-user", "temporal-reasoning"

Design notes
------------
* context_ids  = chat-template prefix + formatted conversation history
* question_ids = question text + chat-template suffix (assistant opening)
* Prefill:    model.model(context_ids, cache)  inside  with press(model):
              → KVzip performs chunked reconstruction and sets masked_key_indices
* Generate:   manual greedy decode with position_ids, matching pipeline.py pattern
* Multi-GPU:  torch.multiprocessing.spawn, each process owns one GPU + one data shard
              → results written to per-GPU JSON shards, aggregated by rank-0
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from string import punctuation

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import kvpress  # noqa: F401  – triggers patch_attention_functions() on import
from kvpress import KVzipPress

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_sessions(haystack_sessions: list) -> str:
    """
    Serialise a list of conversation sessions into a plain-text block.

    Each session is a list of turns, where each turn is a dict with keys
    'role', 'content', and optionally 'date'.
    """
    blocks = []
    for i, session in enumerate(haystack_sessions):
        lines = [f"[Session {i + 1}]"]
        for turn in session:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            date = turn.get("date", "")
            prefix = f"({date}) " if date else ""
            lines.append(f"{role}: {prefix}{content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def tokenize_context_and_question(
    tokenizer,
    sample: dict,
    max_context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build and tokenize context_ids / question_ids for a LongMemEval sample.

    Uses the same separator trick as pipeline.py so that the split point
    between 'context' and 'question' is exact even after chat-template
    formatting.

    Returns
    -------
    context_ids  : (1, ctx_len)  – history up to (but not including) the question
    question_ids : (1, q_len)    – question + chat-template closing / assistant opening
    """
    history_text = format_sessions(sample["haystack_sessions"])
    question = sample["question"]

    # A separator that is unlikely to appear in the text and is easy to split on.
    # We embed it between history and question inside a single user message so
    # that the chat-template prefix (e.g. "<|im_start|>user\n") is included in
    # context_ids.  KVzip's prefix_length is computed the same way (single user
    # message without a system message), so the two align correctly.
    separator = "<<<SEP_KVZIP>>>"
    user_content = history_text + "\n\n" + separator + question

    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    context_text, rest = full_text.split(separator, maxsplit=1)
    # rest = "{question}<|im_end|>\n<|im_start|>assistant\n"

    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False)
    question_ids = tokenizer.encode(rest, return_tensors="pt", add_special_tokens=False)

    if context_ids.shape[1] > max_context_length:
        logger.warning(
            f"Context truncated from {context_ids.shape[1]} to {max_context_length} tokens "
            f"(sample {sample.get('question_id', '?')})."
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

    Mirrors pipeline.py _forward + generate_answer, but without the
    Pipeline wrapper.

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

    # ── 1. Prefill (KVzip compression happens inside the context manager) ─────
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
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in punctuation)
    return " ".join(text.split())


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    """Token-level F1 (max over all ground truths)."""
    pred_tokens = _normalize(prediction).split()
    best = 0.0
    for gt in ground_truths:
        gt_tokens = _normalize(gt).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        p = len(common) / len(pred_tokens) if pred_tokens else 0.0
        r = len(common) / len(gt_tokens) if gt_tokens else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        best = max(best, f1)
    return best


def compute_em(prediction: str, ground_truths: list[str]) -> float:
    pred = _normalize(prediction)
    return float(any(pred == _normalize(gt) for gt in ground_truths))


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
        qid = sample.get("question_id", "?")
        try:
            context_ids, question_ids = tokenize_context_and_question(
                tokenizer, sample, args.max_context_length
            )
            prediction = run_inference(
                model=model,
                press=press,
                context_ids=context_ids,
                question_ids=question_ids,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            log.error(f"Sample {qid} failed: {exc}", exc_info=True)
            prediction = ""

        # Normalise ground truth to a list
        ground_truths = sample.get("expected_answer", [])
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        results.append({
            "question_id": qid,
            "question_type": sample.get("question_type", ""),
            "question": sample.get("question", ""),
            "expected_answer": ground_truths,
            "predicted_answer": prediction,
            "context_tokens": context_ids.shape[1],
            "f1": compute_f1(prediction, ground_truths),
            "em": compute_em(prediction, ground_truths),
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
    """Merge per-GPU shards, compute metrics, write final JSON."""
    all_results: list[dict] = []
    for rank in range(world_size):
        shard_path = output_path.replace(".json", f"_shard{rank}.json")
        with open(shard_path, encoding="utf-8") as f:
            all_results.extend(json.load(f))

    all_results.sort(key=lambda x: str(x["question_id"]))

    n = len(all_results)
    overall_f1 = sum(r["f1"] for r in all_results) / n
    overall_em = sum(r["em"] for r in all_results) / n

    by_type: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_type[r["question_type"]].append(r)

    metrics = {
        "n_samples": n,
        "f1": overall_f1,
        "em": overall_em,
        "by_type": {
            qt: {
                "n": len(rs),
                "f1": sum(r["f1"] for r in rs) / len(rs),
                "em": sum(r["em"] for r in rs) / len(rs),
            }
            for qt, rs in sorted(by_type.items())
        },
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
        description="LongMemEval evaluation with Qwen3-8B + KVzipPress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset", default="xiaowu0162/longmemeval",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--subset", default=None,
        help="Dataset config/subset name (e.g. 'oracle', 's500', 's200'). "
             "None loads the default config.",
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
        "--max_new_tokens", type=int, default=100,
        help="Maximum tokens to generate per answer",
    )
    parser.add_argument(
        "--max_context_length", type=int, default=131072,
        help="Maximum context token length (truncated if exceeded)",
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
        "--output_dir", default="./results/longmemeval",
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
    log.info(f"Loading {args.dataset} (subset={args.subset}, split={args.split}) ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    samples = ds.to_list()

    if args.fraction < 1.0:
        import random
        random.seed(args.seed)
        k = max(1, int(len(samples) * args.fraction))
        samples = random.sample(samples, k)
        log.info(f"Sampled {k} / {len(ds)} examples (fraction={args.fraction})")

    log.info(f"Evaluating on {len(samples)} samples")

    # ── Output path ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "--")
    subset_tag = f"_{args.subset}" if args.subset else ""
    output_path = str(
        output_dir / f"{model_tag}{subset_tag}_kvzip_cr{args.compression_ratio:.2f}.json"
    )
    log.info(f"Results → {output_path}")

    # ── Spawn workers ─────────────────────────────────────────────────────────
    if world_size == 1:
        # Run inline (avoids multiprocessing overhead for single-GPU runs)
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
    log.info(f"  F1 : {metrics['f1']:.4f}")
    log.info(f"  EM : {metrics['em']:.4f}")
    log.info(f"  N  : {metrics['n_samples']}")
    if metrics["by_type"]:
        log.info("  Per-type breakdown:")
        for qt, m in metrics["by_type"].items():
            log.info(f"    {qt:40s}  F1={m['f1']:.4f}  EM={m['em']:.4f}  n={m['n']}")
    log.info(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
