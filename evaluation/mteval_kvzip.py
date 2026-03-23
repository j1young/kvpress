#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MT-Eval benchmark evaluation with Qwen3-8B + KV press.

Runs inference on the FIRST dialogue of a chosen MT-Eval subset,
executing each turn sequentially with fresh KV compression on the
accumulated multi-turn context.

Usage
-----
# Default: recollection_multi_cls, first row (KVzipPress)
python mteval_kvzip.py

# Different subset
python mteval_kvzip.py --subset expansion_multi

# Adjust compression
python mteval_kvzip.py --compression_ratio 0.3

# Use a different press
python mteval_kvzip.py --press expected_attention
python mteval_kvzip.py --press kvzap
python mteval_kvzip.py --press fastkvzip

Dataset
-------
HuggingFace: wckwan/MT-Eval
  Subsets: recollection_single_cls, recollection_multi_cls,
           recollection_single_global-inst, recollection_multi_global-inst,
           expansion_single, expansion_multi,
           refinement_single, refinement_multi,
           follow-up_single, follow-up_multi

Row schema:
  id   : str        – dialogue ID
  conv : list[dict]  – list of turns, each with:
      user         : str   – user utterance
      sys          : str   – reference system response
      id           : str   – turn ID
      inst         : str   – instruction embedded in user utterance
      do_inference : bool   – whether this turn needs model inference

Design notes
------------
* Multi-turn chat template: each turn → {"role":"user"/"assistant"} message
* Separator trick on the LAST user message to split context_ids / question_ids
* Prefill:    model.model(context_ids, cache)  inside  with press(model):
              → KVzip performs chunked reconstruction and sets masked_key_indices
* Generate:   manual greedy decode with position_ids, matching pipeline.py pattern
* Context always uses REFERENCE sys (not generated) to avoid error propagation,
  matching MT-Eval's evaluation methodology
* Scoring:
    - cls subsets        → containment (ref_answer in generated), scaled 0–10
    - other subsets      → ROUGE-L recall × 100
"""

import argparse
import json
import logging
import re
import string
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import kvpress  # noqa: F401  – triggers patch_attention_functions() on import
from kvpress import ExpectedAttentionPress, FastKVzipPress, KVzapPress, KVzipPress

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_messages(history: list[dict], current_user_msg: str) -> list[dict]:
    """
    Build a multi-turn messages list from conversation history.

    Parameters
    ----------
    history : list[dict]
        Previous turns, each with 'user' and 'sys' keys.
    current_user_msg : str
        The current user utterance (to be answered by the model).

    Returns
    -------
    list[dict]
        Messages in the format expected by tokenizer.apply_chat_template.
    """
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["sys"]})
    messages.append({"role": "user", "content": current_user_msg})
    return messages


def tokenize_for_turn(
    tokenizer,
    history: list[dict],
    current_user_msg: str,
    max_context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize accumulated multi-turn context + current user message into
    context_ids / question_ids using the separator trick.

    KVzip compatibility:
      KVzip's prefix_length = len("<|im_start|>user\\n") for Qwen3.
      Our context_ids starts with exactly that prefix, so the alignment
      is correct.  The multi-turn special tokens (role tags between turns)
      are part of the scored content, which is fine – KVzip reconstructs
      them just like normal text tokens.

    Returns
    -------
    context_ids  : (1, ctx_len)
    question_ids : (1, q_len)  – just the closing/assistant-opening tokens
    """
    separator = "<<<SEP_MTEVAL>>>"
    messages = build_messages(history, current_user_msg + separator)

    full_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    context_text, question_suffix = full_text.split(separator, maxsplit=1)
    # context_text   = "...<|im_start|>user\n{current_user_msg}"
    # question_suffix = "<|im_end|>\n<|im_start|>assistant\n"

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
    Prefill with KVzip compression then greedily decode the answer.
    Mirrors pipeline.py _forward + generate_answer.
    """
    device = next(model.parameters()).device
    context_ids = context_ids.to(device)
    question_ids = question_ids.to(device)
    context_length = context_ids.shape[1]

    cache = DynamicCache()

    # ── 1. Prefill (KVzip compression happens at context manager exit) ────────
    with press(model):
        model.model(
            input_ids=context_ids,
            past_key_values=cache,
        )

    # ── 2. Process question (closing/assistant-opening) tokens ────────────────
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

def _remove_punctuations(text: str) -> str:
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text


def score_cls(generated: str, reference: str) -> float:
    """
    Containment scoring for classification (recollection_*_cls).
    Matches MT-Eval's evaluate_recollection_cls: ref in gen → score × 10.
    """
    gen = _remove_punctuations(generated).lower().strip()
    ref = _remove_punctuations(reference).lower().strip()
    return 10.0 if ref in gen else 0.0


def _rouge_l_recall(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L recall (longest common subsequence / len(reference)).
    Simple implementation without external dependencies.
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not ref_tokens:
        return 1.0
    if not hyp_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    # LCS via DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    lcs = prev[n]
    return lcs / m


def score_open(generated: str, reference: str) -> float:
    """ROUGE-L recall × 100, for expansion / refinement / follow-up tasks."""
    return round(_rouge_l_recall(generated, reference) * 100, 2)


def get_scorer(subset: str | None):
    """Return the appropriate scoring function based on the subset name."""
    if subset is not None and "cls" in subset:
        return score_cls
    return score_open


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MT-Eval single-dialogue evaluation with Qwen3-8B + KVzipPress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset", default="wckwan/MT-Eval",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--subset", default=None,
        help="Dataset config/subset name",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--row_index", type=int, default=0,
        help="Which dialogue row to evaluate (0-indexed)",
    )
    parser.add_argument(
        "--press", default="kvzip",
        choices=["kvzip", "expected_attention", "kvzap", "fastkvzip"],
        help="KV press algorithm to use",
    )
    parser.add_argument(
        "--compression_ratio", type=float, default=0.5,
        help="Fraction of KV pairs to prune",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096,
        help="Maximum tokens to generate per turn",
    )
    parser.add_argument(
        "--max_context_length", type=int, default=131072,
        help="Maximum context token length",
    )
    parser.add_argument(
        "--device", default=None,
        help="CUDA device (e.g. 'cuda:0'). Defaults to cuda:0 if available.",
    )
    parser.add_argument(
        "--output_dir", default="./results/mteval",
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

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wall_start = time.perf_counter()

    # ── Load dataset ──────────────────────────────────────────────────────────
    local_path = Path(args.dataset)
    if local_path.exists() and local_path.is_file():
        log.info(f"Loading local file {args.dataset} ...")
        with open(local_path, encoding="utf-8") as f:
            sample = json.load(f)
    else:
        if args.subset is None:
            parser_error = "--subset is required when loading from HuggingFace (e.g. recollection_multi_cls)"
            raise ValueError(parser_error)
        log.info(f"Loading {args.dataset} (subset={args.subset}, split={args.split}) ...")
        ds = load_dataset(args.dataset, args.subset, split=args.split)
        sample = ds[args.row_index]
    conv = sample["conv"]
    dialogue_id = sample["id"]
    n_turns = len(conv)
    n_inference_turns = sum(1 for t in conv if t["do_inference"])

    log.info(
        f"Dialogue '{dialogue_id}': {n_turns} turns total, "
        f"{n_inference_turns} inference turns"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    press_map = {
        "kvzip": KVzipPress,
        "expected_attention": ExpectedAttentionPress,
        "kvzap": KVzapPress,
        "fastkvzip": FastKVzipPress,
    }
    press = press_map[args.press](compression_ratio=args.compression_ratio)
    subset_tag = args.subset if args.subset is not None else Path(args.dataset).stem
    scorer = get_scorer(subset_tag)

    # ── Sequential turn execution ─────────────────────────────────────────────
    history: list[dict] = []  # accumulated reference turns
    results: list[dict] = []
    inference_turn_idx = 0

    log.info("\n" + "=" * 80)
    log.info(f"Dialogue: {dialogue_id}  |  Subset: {subset_tag}")
    log.info("=" * 80)

    for turn_idx, turn in enumerate(conv):
        user_msg = turn["user"]
        ref_sys = turn["sys"]
        do_inference = turn["do_inference"]

        log.info(f"\n--- Turn {turn_idx + 1}/{n_turns} (id={turn['id']}) ---")
        log.info(f"User: {user_msg[:200]}{'...' if len(user_msg) > 200 else ''}")

        if not do_inference:
            log.info(f"[skip inference] Using reference response.")
            log.info(f"Ref:  {ref_sys[:200]}{'...' if len(ref_sys) > 200 else ''}")
            history.append({"user": user_msg, "sys": ref_sys})
            continue

        # ── Inference turn ────────────────────────────────────────────────────
        inference_turn_idx += 1

        context_ids, question_ids = tokenize_for_turn(
            tokenizer, history, user_msg, args.max_context_length
        )
        ctx_tokens = context_ids.shape[1]
        log.info(f"Context tokens: {ctx_tokens}")

        try:
            with torch.inference_mode():
                prediction = run_inference(
                    model=model,
                    press=press,
                    context_ids=context_ids,
                    question_ids=question_ids,
                    max_new_tokens=args.max_new_tokens,
                    tokenizer=tokenizer,
                )
        except Exception as exc:
            log.error(f"Inference failed: {exc}", exc_info=True)
            prediction = ""

        score = scorer(prediction, ref_sys)

        log.info(f"Gen:  {prediction[:300]}{'...' if len(prediction) > 300 else ''}")
        log.info(f"Ref:  {ref_sys[:300]}{'...' if len(ref_sys) > 300 else ''}")
        log.info(f"Score: {score}")

        results.append({
            "dialogue_id": dialogue_id,
            "turn_id": turn["id"],
            "turn_index": turn_idx,
            "inference_turn": inference_turn_idx,
            "user": user_msg,
            "reference": ref_sys,
            "prediction": prediction,
            "context_tokens": ctx_tokens,
            "score": score,
        })

        # Always use REFERENCE sys for subsequent context (no error propagation)
        history.append({"user": user_msg, "sys": ref_sys})
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_sec = time.perf_counter() - wall_start

    if results:
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores)

        log.info("\n" + "=" * 80)
        log.info("SUMMARY")
        log.info("=" * 80)
        log.info(f"Dialogue     : {dialogue_id}")
        log.info(f"Subset       : {subset_tag}")
        log.info(f"Press        : {args.press}")
        log.info(f"Compression  : {args.compression_ratio}")
        log.info(f"Turns scored : {len(results)}")
        log.info(f"Avg score    : {avg_score:.2f}")
        log.info(f"Total time   : {elapsed_sec:.1f}s ({elapsed_sec / 60:.1f}min)")
        log.info("Per-turn scores:")
        for r in results:
            log.info(
                f"  Turn {r['inference_turn']:2d} (idx={r['turn_index']:2d}): "
                f"score={r['score']:6.2f}  ctx_tokens={r['context_tokens']}"
            )
    else:
        avg_score = 0.0
        log.info("No inference turns found in this dialogue.")
        log.info(f"Total time   : {elapsed_sec:.1f}s ({elapsed_sec / 60:.1f}min)")

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace("/", "--")
    output_path = (
        output_dir / f"{model_tag}_{subset_tag}_row{args.row_index}_{args.press}_cr{args.compression_ratio:.2f}.json"
    )

    output_data = {
        "config": {
            "model": args.model,
            "subset": subset_tag,
            "row_index": args.row_index,
            "press": args.press,
            "compression_ratio": args.compression_ratio,
            "max_new_tokens": args.max_new_tokens,
        },
        "dialogue_id": dialogue_id,
        "n_turns": n_turns,
        "n_inference_turns": len(results),
        "avg_score": avg_score,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
