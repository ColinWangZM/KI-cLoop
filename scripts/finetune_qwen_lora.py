#!/usr/bin/env python3
"""Fine-tune the organic-synthesis adapter without machine-specific paths."""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


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


def build_preprocessor(tokenizer, max_length: int):
    def preprocess(example):
        user_text = f"{example.get('instruction', '')}{example.get('input', '')}"
        prompt_ids = tokenizer(
            render_prompt(tokenizer, user_text), add_special_tokens=False
        )["input_ids"]
        response_text = str(example.get("output", "")) + (tokenizer.eos_token or "")
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + response_ids)[:max_length]
        prompt_length = min(len(prompt_ids), max_length)
        labels = ([-100] * prompt_length + response_ids)[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="JSON records with instruction/input/output fields.")
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_QWEN_MODEL_PATH", "Qwen/Qwen2-7B-Instruct"),
    )
    parser.add_argument("--output-dir", default="checkpoints/or_lora")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_json(args.dataset)
    required = {"instruction", "input", "output"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_pandas(frame, preserve_index=False)
    tokenized = dataset.map(
        build_preprocessor(tokenizer, args.max_length),
        remove_columns=dataset.column_names,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            num_train_epochs=args.epochs,
            save_steps=args.save_steps,
            learning_rate=args.learning_rate,
            save_on_each_node=True,
            gradient_checkpointing=True,
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()


if __name__ == "__main__":
    main()
