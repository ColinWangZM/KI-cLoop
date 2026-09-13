#!/usr/bin/env python3
"""Aggregate evaluation results and export task-level averages in percentage scale.

This script walks through the ``llm_eval_results`` directory, loads any
``evaluations.json`` files, and writes the aggregated averages per
``(question_type, model, dataset)`` trio to a CSV file. Each score is converted
from the original 0–10 scale to percentage (0–100). Optionally, detailed per-
question rows can be exported to a separate CSV.

Usage
-----
python collect_evaluations.py [--eval-root PATH] [--output PATH] [--detailed-output PATH]

Options
-------
--eval-root        Base directory that contains evaluation subfolders. Defaults
                   to "llm_eval_results" relative to this script.
--output           CSV path for aggregated averages. Defaults to
                   "evaluation_scores.csv" in the project root.
--detailed-output  Optional CSV path for detailed per-question scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect evaluation scores and export to CSV.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(__file__).resolve().parent / "llm_eval_results",
        help="Root directory that contains evaluation outputs (default: llm_eval_results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_scores.csv",
        help="Destination CSV for aggregated task averages (default: evaluation_scores.csv).",
    )
    parser.add_argument(
        "--detailed-output",
        type=Path,
        help="Optional destination CSV for detailed per-question scores.",
    )
    return parser.parse_args()


def find_evaluation_files(eval_root: Path) -> Iterable[Path]:
    """Yield all evaluation JSON files under ``eval_root``."""
    if not eval_root.exists():
        return []
    return eval_root.rglob("evaluations.json")


def load_records(file_path: Path, project_root: Path) -> List[Dict[str, Union[str, float, None]]]:
    """Load evaluation records from a single JSON file."""
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file {file_path}: {exc}") from exc

    question_type = payload.get("question_type")
    model_name = payload.get("model")
    records = []

    for record in payload.get("records", []):
        raw_score = record.get("score")
        if raw_score is None:
            continue  # Skip entries without a score

        score_percent = round(float(raw_score) * 10.0, 2)
        try:
            relative_path = file_path.relative_to(project_root)
        except ValueError:
            relative_path = file_path.name

        records.append(
            {
                "question_type": record.get("question_type") or question_type,
                "model": model_name,
                "dataset": record.get("dataset"),
                "question_id": record.get("question_id"),
                "score_percent": score_percent,
                "source_file": str(relative_path),
            }
        )
    return records


def write_detailed_csv(rows: Iterable[Dict[str, Union[str, float, None]]], output_path: Path) -> None:
    """Write detailed per-question rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_type", "model", "dataset", "question_id", "score_percent", "source_file"]

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(rows: Iterable[Dict[str, Union[str, float, None]]], output_path: Path) -> None:
    """Write aggregated average scores per (question_type, model, dataset)."""
    aggregates: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = (row.get("question_type"), row.get("model"), row.get("dataset"))
        score_value = row.get("score_percent")
        try:
            score = float(score_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        aggregates[key].append(score)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_type", "model", "dataset", "average_score_percent", "num_questions"]

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (question_type, model_name, dataset), scores in sorted(aggregates.items()):
            if not scores:
                continue
            average_score = round(sum(scores) / len(scores), 2)
            writer.writerow(
                {
                    "question_type": question_type,
                    "model": model_name,
                    "dataset": dataset,
                    "average_score_percent": average_score,
                    "num_questions": len(scores),
                }
            )


def main() -> None:
    args = parse_args()
    evaluation_files = list(find_evaluation_files(args.eval_root))

    if not evaluation_files:
        raise FileNotFoundError(f"No evaluations.json files found under {args.eval_root}")

    project_root = Path(__file__).resolve().parent

    all_rows: List[Dict[str, Union[str, float, None]]] = []
    for file_path in evaluation_files:
        all_rows.extend(load_records(file_path, project_root))

    if not all_rows:
        raise ValueError("No valid score records were found in the evaluation files.")

    write_summary_csv(all_rows, args.output)

    if args.detailed_output:
        write_detailed_csv(all_rows, args.detailed_output)


if __name__ == "__main__":
    main()
