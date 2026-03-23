#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regenerate the 'sys' field of a merged MT-Eval JSON using actual Qwen3-8B outputs.

Each turn is run sequentially; the model's generated output becomes the context
for the next turn (no reference leakage). Turns with do_inference=False can be
optionally skipped (keeping their original reference sys).

Usage
-----
# Regenerate only do_inference=True turns (default)
python generate_sys.py --input merged_follow-up_multi.json

# Regenerate ALL turns
python generate_sys.py --input merged_follow-up_multi.json --all_turns

# Custom output path
python generate_sys.py --input merged_follow-up_multi.json --output regen_follow-up_multi.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_messages(history: list[dict], current_user_msg: str) -> list[dict]:
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["sys"]})
    messages.append({"role": "user", "content": current_user_msg})
    return messages


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    history: list[dict],
    user_msg: str,
    max_new_tokens: int,
    max_context_length: int,
) -> str:
    """Build the full prompt and generate a response with model.generate()."""
    messages = build_messages(history, user_msg)

    template_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    # enable_thinking is Qwen3-specific; skip for other models
    chat_template_src = getattr(tokenizer, "chat_template", "") or ""
    if "enable_thinking" in chat_template_src:
        template_kwargs["enable_thinking"] = False

    result = tokenizer.apply_chat_template(messages, **template_kwargs)

    input_ids = result["input_ids"]
    
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if input_ids.shape[1] > max_context_length:
        logger.warning(
            f"Context truncated from {input_ids.shape[1]} to {max_context_length} tokens."
        )
        input_ids = input_ids[:, -max_context_length:]

    input_ids = input_ids.to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate sys fields in a merged MT-Eval JSON with Qwen3-8B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Input merged JSON file (e.g. merged_follow-up_multi.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file. Defaults to <input_stem>_generated.json",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--all_turns", action="store_true", default=False,
        help="Generate for ALL turns. Default: only do_inference=True turns.",
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
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wall_start = time.perf_counter()

    # ── Load input ────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    conv: list[dict] = data["conv"]
    dialogue_id: str = data["id"]
    n_turns = len(conv)
    n_target = sum(1 for t in conv if args.all_turns or t["do_inference"])
    log.info(f"Loaded '{dialogue_id}': {n_turns} turns, {n_target} to generate")

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

    # ── Sequential generation ─────────────────────────────────────────────────
    history: list[dict] = []  # uses model-generated sys (not reference)
    updated_conv: list[dict] = []
    n_generated = 0

    log.info("\n" + "=" * 80)
    log.info(f"Dialogue: {dialogue_id}  |  all_turns={args.all_turns}")
    log.info("=" * 80)

    for turn_idx, turn in enumerate(conv):
        user_msg = turn["user"]
        ref_sys = turn["sys"]
        do_infer = turn["do_inference"]

        log.info(f"\n--- Turn {turn_idx + 1}/{n_turns} (id={turn['id']}, do_inference={do_infer}) ---")
        log.info(f"User: {user_msg[:200]}{'...' if len(user_msg) > 200 else ''}")

        should_generate = args.all_turns or do_infer

        if not should_generate:
            log.info("[skip] Keeping reference sys.")
            log.info(f"Ref:  {ref_sys[:200]}{'...' if len(ref_sys) > 200 else ''}")
            updated_turn = dict(turn)
            history.append({"user": user_msg, "sys": ref_sys})
        else:
            try:
                generated = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    history=history,
                    user_msg=user_msg,
                    max_new_tokens=args.max_new_tokens,
                    max_context_length=args.max_context_length,
                )
            except Exception as exc:
                log.error(f"Generation failed: {exc}", exc_info=True)
                generated = ref_sys  # fall back to reference on error

            n_generated += 1
            log.info(f"Gen:  {generated[:300]}{'...' if len(generated) > 300 else ''}")

            updated_turn = dict(turn)
            updated_turn["sys"] = generated
            updated_turn["ref_sys"] = ref_sys  # preserve original reference
            history.append({"user": user_msg, "sys": generated})
            torch.cuda.empty_cache()

        updated_conv.append(updated_turn)

    # ── Save ──────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - wall_start
    output_path = Path(args.output or input_path.stem + "_generated.json")

    output_data = {
        "id": dialogue_id,
        "config": {
            "model": args.model,
            "all_turns": args.all_turns,
            "max_new_tokens": args.max_new_tokens,
            "n_generated": n_generated,
            "elapsed_sec": round(elapsed, 1),
        },
        "conv": updated_conv,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log.info("\n" + "=" * 80)
    log.info("DONE")
    log.info("=" * 80)
    log.info(f"Turns generated : {n_generated} / {n_turns}")
    log.info(f"Total time      : {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    log.info(f"Saved to        : {output_path}")


if __name__ == "__main__":
    main()
