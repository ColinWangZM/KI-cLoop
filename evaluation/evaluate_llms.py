"""Usage
---------
Collect answers without judging:
    python evaluate_llms.py --dataset-dir Q1-Q6 --models "openai:gpt-5.5,qwen:qwen3.7-plus,deepseek:deepseek-v4-pro"

Run only selected question types (comma-separated):
    python evaluate_llms.py --question-types "Chain Reasoning,Multi-path Comparison"

Outputs stream each model's answer to stdout while saving the full JSON report.
Per-question JSON files are organized under <output-stem>/<QuestionType>/<Model>.
Tasks without reference answers (e.g., open-ended ontology questions) are included during collection.

Environment variables:
    OPENAI_API_KEY for openai:* models
    DASHSCOPE_API_KEY for qwen:* models and other DashScope-compatible baselines
    DEEPSEEK_API_KEY for deepseek:* models
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from llm_utils import DEFAULT_SYSTEM_PROMPT, build_candidate_messages, create_client, load_tasks


def sanitize_component(value: str, fallback: str) -> str:
    token = (value or "").strip()
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", token)
    return token or fallback


def persist_per_question_records(results: Dict[str, object], base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for descriptor, model_block in results.get("models", {}).items():
        model_component = sanitize_component(descriptor, "model")
        for question_type, records in model_block.get("question_types", {}).items():
            qt_component = sanitize_component(question_type, "Unknown")
            target_dir = base_dir / qt_component / model_component
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "question_type": question_type,
                "model": descriptor,
                "records": records,
            }
            with (target_dir / "responses.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)


def collect_responses(models: List[str], dataset_dir: Path, delay: float, max_output_tokens: int,
                      temperature: float, question_types: Optional[List[str]]) -> Dict[str, object]:
    tasks = load_tasks(dataset_dir, require_reference=False)
    if question_types:
        allowed = {qt.strip() for qt in question_types if qt.strip()}
        tasks = [task for task in tasks if (task.question_type or "") in allowed]
    if not tasks:
        raise RuntimeError("No tasks with reference answers found. Ensure CSV files include 'Answer' values.")

    candidate_clients = {descriptor: create_client(descriptor, temperature=temperature) for descriptor in models}

    results: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "parameters": {
            "delay": delay,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        },
        "models": {},
    }

    for descriptor in candidate_clients:
        results["models"][descriptor] = {
            "question_types": {},
        }

    for idx, task in enumerate(tasks, start=1):
        question_type = task.question_type or "Unknown"
        for descriptor, client in candidate_clients.items():
            model_bucket = results["models"][descriptor]
            qt_bucket = model_bucket["question_types"].setdefault(question_type, [])
            record = {
                "dataset": task.dataset,
                "question_id": task.question_id,
                "question": task.question,
                "selection": task.selection,
                "reference_answer": task.reference_answer,
                "correct_choice": task.correct_choice,
                "answer": "",
            }
            try:
                messages = build_candidate_messages(task, DEFAULT_SYSTEM_PROMPT)
                candidate_answer = client.complete(messages, max_output_tokens=max_output_tokens)
                record["answer"] = candidate_answer
                print(f"[{descriptor}] {task.question_id or 'unknown'} => {candidate_answer}", flush=True)
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)
                print(f"[{descriptor}] {task.question_id or 'unknown'} ERROR: {exc}", flush=True)
            qt_bucket.append(record)
            time.sleep(delay)
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(tasks)} questions for all models.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect responses from multiple LLMs without judging.")
    parser.add_argument("--dataset-dir", default="Q1-Q6", help="Path to the directory containing CSV datasets.")
    parser.add_argument(
        "--models",
        default="openai:gpt-5.5,qwen:qwen3.7-plus,deepseek:deepseek-v4-pro",
        help="Comma-separated list of model descriptors (provider:model).",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between API calls.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for candidate models.")
    parser.add_argument("--max-output-tokens", type=int, default=512, help="Max output tokens per model response.")
    parser.add_argument("--output", default="model_answers.json", help="Where to store collected answers.")
    parser.add_argument(
        "--question-types",
        default="",
        help="Optional comma-separated list of question types to include (matches CSV QuestionType).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
    model_list = [item.strip() for item in args.models.split(",") if item.strip()]
    if not model_list:
        raise ValueError("At least one model descriptor must be provided.")

    question_types = [item.strip() for item in args.question_types.split(",") if item.strip()]

    results = collect_responses(
        models=model_list,
        dataset_dir=dataset_dir,
        delay=args.delay,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        question_types=question_types if question_types else None,
    )

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    per_question_dir = output_path.with_suffix("")
    persist_per_question_records(results, per_question_dir)

    total_answers = 0
    for descriptor, model_data in results["models"].items():
        model_total = sum(len(records) for records in model_data["question_types"].values())
        total_answers += model_total
        print(f"Model {descriptor}: captured {model_total} answers across {len(model_data['question_types'])} question types.")

    print(f"Collected {total_answers} answers across {len(results['models'])} models.")
    print(f"Detailed responses saved to {output_path}")
    print(f"Per-question JSON saved under {per_question_dir}")


if __name__ == "__main__":
    main()
