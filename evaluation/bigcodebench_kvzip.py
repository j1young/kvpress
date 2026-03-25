#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BigCodeBench-Hard benchmark evaluation with Qwen3-8B + KV press.

Generates Python code from instruct_prompt, then scores by executing
the generated code against the provided unittest test suite.

Usage
-----
# Default: KVzipPress, compression_ratio=0.5
python bigcodebench_kvzip.py

# Multiple GPUs
python bigcodebench_kvzip.py --n_gpus 4

# Different press / compression
python bigcodebench_kvzip.py --press expected_attention --compression_ratio 0.3

Dataset
-------
HuggingFace: bigcode/bigcodebench-hard
  Splits: v0.1.0_hf, v0.1.1, v0.1.2, v0.1.3, v0.1.4

Columns:
  task_id          : str  – unique task identifier (e.g. "BigCodeBench/13")
  instruct_prompt  : str  – natural language instruction
  test             : str  – unittest.TestCase code for validation
  entry_point      : str  – always "task_func"
  canonical_solution : str – reference implementation
  libs             : list[str] – required libraries

Scoring
-------
  Pass@1: fraction of tasks where ALL unit tests pass on the generated code.

Design notes
------------
* System prompt instructs model to output only Python code defining task_func.
* enable_thinking=True for Qwen3 — model reasons before answering.
* Post-processing strips <think>...</think> block, then extracts Python code.
* Tests executed in an isolated subprocess with timeout.
* Separator trick on user message to split context_ids / question_ids.
* Prefill:    model.model(context_ids, cache)  inside  with press(model):
* Generate:   manual greedy decode with position_ids
* Multi-GPU:  torch.multiprocessing.spawn with round-robin sharding.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
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
    "Write a complete Python function named `task_func` that solves the given task. "
    "Output ONLY the Python code (including any necessary imports). "
    "Do not include explanations, examples, or test code."
)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_prompt(
    tokenizer,
    instruct_prompt: str,
    max_context_length: int,
    enable_thinking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize instruct_prompt into context_ids / question_ids using the
    separator trick (same pattern as ruler_kvzip.py).

    context_ids  = chat template prefix + system msg + instruct_prompt
    question_ids = chat template suffix (generation prompt tokens)
    """
    separator = "<<<SEP_BIGCODEBENCH>>>"

    full_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruct_prompt + separator},
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
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
    3. Greedy decode until EOS or max_new_tokens
    """
    device = next(model.parameters()).device
    context_ids = context_ids.to(device)
    question_ids = question_ids.to(device)
    context_length = context_ids.shape[1]

    cache = DynamicCache()

    # ── 1. Prefill (press compression happens at context manager exit) ────────
    if press is not None:
        with press(model):
            model.model(
                input_ids=context_ids,
                past_key_values=cache,
            )
    else:
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
# Code extraction and test execution
# ──────────────────────────────────────────────────────────────────────────────

def strip_thinking(raw_output: str) -> str:
    """Remove <think>...</think> block from Qwen3 thinking output."""
    return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()


def extract_code(raw_output: str) -> str:
    """
    Extract Python code from model output.

    1. Strip thinking tokens.
    2. Look for markdown code blocks (```python ... ``` or ``` ... ```).
    3. Fall back to raw output if no code blocks found.
    """
    text = strip_thinking(raw_output)

    # Try to find fenced code blocks
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    return text.strip()


def execute_tests(
    generated_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 30,
) -> dict:
    """
    Execute generated code + test code in an isolated subprocess.

    Returns
    -------
    dict with keys: passed (bool), error (str | None), returncode (int)
    """
    combined = f"""\
import unittest
import sys

# --- Generated code ---
{generated_code}

# --- Test code ---
{test_code}

if __name__ == "__main__":
    unittest.main(argv=[''], exit=True, verbosity=0)
"""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(combined)
            tmp_path = tmp.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        error = result.stderr.strip() if not passed else None
        return {"passed": passed, "error": error, "returncode": result.returncode}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout after {timeout}s", "returncode": -1}
    except Exception as exc:
        return {"passed": False, "error": str(exc), "returncode": -1}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def score_sample(
    generated_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 30,
) -> tuple[float, dict]:
    """Return (score, exec_result). Score is 1.0 if all tests pass, 0.0 otherwise."""
    result = execute_tests(generated_code, test_code, entry_point, timeout)
    return (1.0 if result["passed"] else 0.0, result)


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
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    press_map = {
        "kvzip": KVzipPress,
        "expected_attention": ExpectedAttentionPress,
        "kvzap": KVzapPress,
        "fastkvzip": FastKVzipPress,
    }
    if args.compression_ratio == 0.0:
        press = None
        log.info("compression_ratio=0 → skipping press (no KV compression)")
    else:
        press = press_map[args.press](compression_ratio=args.compression_ratio)

    # Round-robin shard
    shard = samples[rank::world_size]
    log.info(f"Processing {len(shard)} / {len(samples)} samples ...")

    results = []
    for sample in tqdm(shard, desc=f"GPU{rank}", position=rank, leave=True):
        task_id = sample["task_id"]
        instruct_prompt = sample["instruct_prompt"]
        test_code = sample["test"]
        entry_point = sample["entry_point"]

        try:
            context_ids, question_ids = tokenize_prompt(
                tokenizer,
                instruct_prompt,
                args.max_context_length,
                enable_thinking=args.enable_thinking,
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

        generated_code = extract_code(raw_output)
        score, exec_result = score_sample(
            generated_code, test_code, entry_point, args.timeout,
        )

        log.info(
            f"{task_id}: score={score:.0f}  "
            f"ctx_tokens={context_ids.shape[1]}  "
            f"{'PASS' if exec_result['passed'] else 'FAIL'}"
        )
        if exec_result.get("error"):
            log.debug(f"  Error: {exec_result['error'][:300]}")

        results.append({
            "task_id": task_id,
            "context_tokens": context_ids.shape[1],
            "generated_code": generated_code,
            "raw_output": raw_output[:2000],
            "score": score,
            "passed": exec_result["passed"],
            "error": exec_result.get("error"),
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
    """Merge per-GPU shards and compute Pass@1."""
    all_results: list[dict] = []
    for rank in range(world_size):
        shard_path = output_path.replace(".json", f"_shard{rank}.json")
        with open(shard_path, encoding="utf-8") as f:
            all_results.extend(json.load(f))

    n_total = len(all_results)
    n_passed = sum(1 for r in all_results if r["passed"])
    pass_at_1 = round(n_passed / n_total * 100, 2) if n_total > 0 else 0.0

    metrics = {
        "pass_at_1": pass_at_1,
        "n_tasks": n_total,
        "n_passed": n_passed,
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
        description="BigCodeBench-Hard evaluation with Qwen3-8B + KV press",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset", default="bigcode/bigcodebench-hard",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--split", default="v0.1.4",
        help="Dataset split/version",
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
        "--max_context_length", type=int, default=16384,
        help="Maximum context token length",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=16384,
        help="Maximum tokens to generate (includes thinking tokens)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Timeout in seconds for test execution per task",
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
        "--output_dir", default="./results/bigcodebench",
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
        help="Enable thinking mode for Qwen3 (--enable_thinking / --no-enable_thinking)",
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
    log.info(f"Loading {args.dataset} (split={args.split}) ...")
    ds = load_dataset(args.dataset, split=args.split)
    samples = ds.to_list()

    if args.fraction < 1.0:
        import random
        random.seed(args.seed)
        k = max(1, int(len(samples) * args.fraction))
        samples = random.sample(samples, k)
        log.info(f"Sampled {k} / {len(ds)} examples (fraction={args.fraction})")

    log.info(f"Evaluating {len(samples)} tasks")

    # ── Output path ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "--")
    thinking_tag = "think" if args.enable_thinking else "nothink"
    output_path = str(
        output_dir / f"{model_tag}_{args.split}_{args.press}_cr{args.compression_ratio:.2f}_{thinking_tag}.json"
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
    log.info(f"  Press          : {args.press}")
    log.info(f"  Compression    : {args.compression_ratio}")
    log.info(f"  Thinking       : {args.enable_thinking}")
    log.info(f"  Pass@1         : {metrics['pass_at_1']:.2f}%")
    log.info(f"  Passed / Total : {metrics['n_passed']} / {metrics['n_tasks']}")
    log.info(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
