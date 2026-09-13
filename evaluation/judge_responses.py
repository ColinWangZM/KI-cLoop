"""Usage
---------
Judge previously collected answers:
    python judge_responses.py --answers model_answers.json --judge-model openai:gpt-4.1-mini

Judge a single model and question type:
    python judge_responses.py --answers model_answers.json --judge-model openai:gpt-4.1-mini \
        --models "qwen:qwen-plus" --question-types "Attribute Query"

Auto-discover answers under <QuestionType>/<Model>/responses.json hierarchy:
    python judge_responses.py --answers-root model_answers --judge-model openai:gpt-4.1-mini

Environment variables:
    OPENAI_API_KEY for openai:* judge models
    DASHSCOPE_API_KEY for all other judge providers

Per-question evaluations are exported under <output-stem>/<QuestionType>/<Model>.
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm_utils import (
    Task,
    build_judge_messages,
    create_client,
    REFERENCE_OPTIONAL_TYPES,
    parse_judge_output,
)

AUTO_GRADED_TYPES = {
    "Attribute Query",
    "Multi-Attribute Query",
}


def extract_choice(text: str) -> Optional[str]:
    """Return the first detected option letter in the candidate answer."""
    if not text:
        return None
    match = re.search(r"(?i)answer\s*[:：]\s*([A-Z])", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Z])\b", text.upper())
    return match.group(1).upper() if match else None


def sanitize_component(value: str, fallback: str) -> str:
    token = (value or "").strip()
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", token)
    return token or fallback


def persist_per_question_evaluations(results: Dict[str, object], base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for descriptor, model_block in results.get("models", {}).items():
        model_component = sanitize_component(descriptor, "model")
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for record in model_block.get("records", []):
            qt = record.get("question_type", "Unknown")
            grouped.setdefault(qt, []).append(record)

        for question_type, records in grouped.items():
            qt_component = sanitize_component(question_type, "Unknown")
            target_dir = base_dir / qt_component / model_component
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "question_type": question_type,
                "model": descriptor,
                "records": records,
            }
            with (target_dir / "evaluations.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)


def _discover_answers_root() -> Path:
    candidates = [Path("model_answers"), Path("responses"), Path("answers")]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    for top in Path(".").iterdir():
        if not top.is_dir():
            continue
        for sub in top.iterdir():
            if not sub.is_dir():
                continue
            for leaf in sub.iterdir():
                if leaf.is_dir() and (leaf / "responses.json").exists():
                    return top
    raise FileNotFoundError(
        "Unable to locate responses directory automatically. Provide --answers or --answers-root."
    )


def _load_answers_from_root(root: Path) -> Dict[str, object]:
    if not root.exists():
        raise FileNotFoundError(f"Responses directory '{root}' not found.")
    if not root.is_dir():
        raise NotADirectoryError(f"Responses root '{root}' is not a directory.")

    models: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for question_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in question_dir.iterdir() if p.is_dir()):
            payload_path = model_dir / "responses.json"
            if not payload_path.exists():
                continue
            with payload_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            model_name = payload.get("model") or model_dir.name
            question_type = payload.get("question_type") or question_dir.name
            records = payload.get("records", [])

            model_entry = models.setdefault(model_name, {"question_types": {}})
            model_entry["question_types"].setdefault(question_type, []).extend(records)

    return {"models": models}


def _load_answers_payload(
    answer_path: Optional[Path],
    answers_root: Optional[Path],
) -> Tuple[Dict[str, object], Path]:
    if answer_path:
        if not answer_path.exists():
            raise FileNotFoundError(f"Answers file '{answer_path}' not found.")
        with answer_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), answer_path

    root = answers_root or _discover_answers_root()
    return _load_answers_from_root(root), root


def judge_answers(answer_path: Optional[Path], answers_root: Optional[Path],
                  judge_model: str, delay: float,
                  max_output_tokens: int,
                  focus_models: Optional[List[str]],
                  focus_question_types: Optional[List[str]]) -> Dict[str, object]:
    answer_payload, source_path = _load_answers_payload(answer_path, answers_root)

    models_block = answer_payload.get("models", {})
    if not models_block:
        raise RuntimeError("No model answers found in the provided file.")

    judge_client: Optional[object] = None

    source_path = source_path.resolve()

    results: Dict[str, object] = {
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "answers_source": str(source_path),
        "judge_model": judge_model,
        "parameters": {
            "delay": delay,
            "max_output_tokens": max_output_tokens,
        },
        "scoring": {
            "max_score": 10,
        },
        "models": {},
    }

    if source_path.is_file():
        results["answers_file"] = str(source_path)
    else:
        results["answers_dir"] = str(source_path)

    allowed_models = {m.lower() for m in focus_models} if focus_models else None
    allowed_types = {qt for qt in focus_question_types} if focus_question_types else None

    for descriptor, model_block in models_block.items():
        if allowed_models and descriptor.lower() not in allowed_models:
            continue
        question_types = model_block.get("question_types", {})
        if allowed_types:
            question_types = {
                qt: records for qt, records in question_types.items() if qt in allowed_types
            }
            if not question_types:
                continue
        model_result = results["models"].setdefault(
            descriptor,
            {
                "totals": {"evaluated": 0, "score_sum": 0.0},
                "question_types": {},
                "records": [],
            },
        )

        for question_type, records in question_types.items():
            qt_stats = model_result["question_types"].setdefault(
                question_type, {"evaluated": 0, "score_sum": 0.0}
            )

            for record in records:
                candidate_answer = record.get("answer", "")
                error = record.get("error")
                task = Task(
                    dataset=record.get("dataset", ""),
                    question_id=record.get("question_id", ""),
                    question_type=question_type,
                    question=record.get("question", ""),
                    selection=record.get("selection"),
                    reference_answer=record.get("reference_answer"),
                    correct_choice=record.get("correct_choice"),
                )

                evaluation_entry = {
                    "dataset": task.dataset,
                    "question_id": task.question_id,
                    "question_type": question_type,
                    "candidate_answer": candidate_answer,
                    "reference_answer": task.reference_answer,
                    "correct_choice": task.correct_choice,
                    "score": 0.0,
                    "justification": "",
                }

                if error:
                    evaluation_entry["justification"] = f"Skipped due to collection error: {error}"
                    model_result["records"].append(evaluation_entry)
                    print(
                        f"[SKIP] {descriptor} {question_type} {task.question_id}: collection error -> {error}",
                        flush=True,
                    )
                    continue

                auto_grade_applicable = (
                    question_type in AUTO_GRADED_TYPES and (task.correct_choice or "").strip()
                )

                if auto_grade_applicable:
                    # Multiple-choice questions are handled through direct option matching rather than the LLM judge.
                    expected_choice = (task.correct_choice or "").strip().upper()
                    candidate_choice = extract_choice(candidate_answer)
                    if candidate_choice:
                        score = 10.0 if candidate_choice == expected_choice else 0.0
                        evaluation_entry["score"] = score
                        evaluation_entry["justification"] = (
                            f"Auto-graded via option match: model={candidate_choice}, reference={expected_choice}."
                        )
                    else:
                        score = 0.0
                        evaluation_entry["score"] = score
                        evaluation_entry["justification"] = "Auto-grading failed: could not parse option letter."

                    model_result["records"].append(evaluation_entry)
                    model_result["totals"]["evaluated"] += 1
                    qt_stats["evaluated"] += 1
                    model_result["totals"]["score_sum"] += score
                    qt_stats["score_sum"] += score
                    if score == 10:
                        evaluation_entry["justification"] += " Awarded full score of 10."
                    print(
                        f"[AUTO] {descriptor} {question_type} {task.question_id}: "
                        f"score {score:.2f}/10 - {evaluation_entry['justification']}",
                        flush=True,
                    )
                    continue

                if not task.reference_answer and question_type not in REFERENCE_OPTIONAL_TYPES:
                    evaluation_entry["justification"] = "Skipped: missing reference answer."
                    model_result["records"].append(evaluation_entry)
                    print(
                        f"[SKIP] {descriptor} {question_type} {task.question_id}: missing reference answer.",
                        flush=True,
                    )
                    continue

                try:
                    if judge_client is None:
                        judge_client = create_client(judge_model, temperature=0.0)
                    print(
                        f"[JUDGE] {descriptor} {question_type} {task.question_id}: submitting to {judge_model}...",
                        flush=True,
                    )
                    judge_messages = build_judge_messages(task, candidate_answer)
                    judge_reply = judge_client.complete(judge_messages, max_output_tokens=max_output_tokens)
                    evaluation = parse_judge_output(judge_reply)
                    try:
                        score = float(evaluation.get("score", 0))
                    except (TypeError, ValueError):
                        score = 0.0
                    score = max(0.0, min(10.0, score))
                    justification = evaluation.get("justification", "")
                except Exception as exc:  # noqa: BLE001
                    score = 0.0
                    justification = f"Error during judging: {exc}"

                evaluation_entry["score"] = score
                evaluation_entry["justification"] = justification
                model_result["records"].append(evaluation_entry)
                print(
                    f"[JUDGE] {descriptor} {question_type} {task.question_id}: "
                    f"score {score:.2f}/10 - {justification}",
                    flush=True,
                )

                model_result["totals"]["evaluated"] += 1
                qt_stats["evaluated"] += 1
                model_result["totals"]["score_sum"] += score
                qt_stats["score_sum"] += score

                time.sleep(delay)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge pre-collected LLM answers using an LLM judge.")
    parser.add_argument("--answers", default="model_answers.json", help="Path to the collected answers JSON file.")
    parser.add_argument(
        "--answers-root",
        default="",
        help="Directory containing per-question responses (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--judge-model",
        default="openai:gpt-4.1-mini",
        help="Model descriptor for the judge (provider:model).",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between judge requests.")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Max output tokens for judge responses.")
    parser.add_argument("--output", default="llm_eval_results.json", help="Where to store evaluation results.")
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated list of model descriptors to judge (provider:model).",
    )
    parser.add_argument(
        "--question-types",
        default="",
        help="Optional comma-separated list of question types to judge.",
    )
    args = parser.parse_args()

    answer_path: Optional[Path] = None
    if args.answers:
        candidate_path = Path(args.answers)
        if candidate_path.exists():
            answer_path = candidate_path
        elif args.answers != parser.get_default("answers"):
            print(
                f"Warning: answers file '{candidate_path}' not found. Falling back to auto-discovery.",
                flush=True,
            )

    answers_root = Path(args.answers_root) if args.answers_root else None
    focus_models = [item.strip() for item in args.models.split(",") if item.strip()]
    focus_types = [item.strip() for item in args.question_types.split(",") if item.strip()]

    results = judge_answers(
        answer_path=answer_path,
        answers_root=answers_root,
        judge_model=args.judge_model,
        delay=args.delay,
        max_output_tokens=args.max_output_tokens,
        focus_models=focus_models if focus_models else None,
        focus_question_types=focus_types if focus_types else None,
    )

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    per_question_dir = output_path.with_suffix("")
    persist_per_question_evaluations(results, per_question_dir)

    for descriptor, data in results["models"].items():
        totals = data["totals"]
        evaluated = totals["evaluated"]
        average = totals["score_sum"] / evaluated if evaluated else 0.0
        print(f"Model {descriptor}: mean score {average:.2f}/10 over {evaluated} evaluations")
        for question_type, stats in sorted(data["question_types"].items()):
            qt_eval = stats["evaluated"]
            qt_average = stats["score_sum"] / qt_eval if qt_eval else 0.0
            print(f"  {question_type}: mean {qt_average:.2f}/10 across {qt_eval} questions")

    print(f"Judging results written to {output_path}")
    print(f"Per-question evaluations saved under {per_question_dir}")


if __name__ == "__main__":
    main()
