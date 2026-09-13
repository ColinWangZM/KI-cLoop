#!/usr/bin/env python3
"""Run inference with a Qwen base model and an organic-synthesis LoRA adapter."""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are an expert language model for chemistry and materials science, "
    "including organic synthesis, MOFs, coordination polymers, porous "
    "materials, and perovskites."
)


def render_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"System: {SYSTEM_PROMPT}\nUser: {user_text}\nAssistant:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Question or synthesis request.")
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_QWEN_MODEL_PATH", "Qwen/Qwen2-7B-Instruct"),
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("KGRAG_OR_LORA_PATH", "checkpoints/or_lora"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_id=args.adapter).to(device).eval()
    encoded = tokenizer(
        render_prompt(tokenizer, args.prompt), return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    completion = generated[:, encoded.input_ids.shape[1] :]
    print(tokenizer.batch_decode(completion, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
