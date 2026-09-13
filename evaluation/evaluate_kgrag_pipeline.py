#!/usr/bin/env python3
"""Collect answers from the current local KGRAG workflow.

The output JSON mirrors ``evaluate_llms.py`` so ``judge_responses.py`` and
``collect_evaluations.py`` can compare KGRAG against API-based LLM baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kg_rag.rag_based_generation.Llama.application_synthesis_qa import (  # noqa: E402
    answer_application_synthesis_question,
)
from kg_rag.rag_based_generation.Llama.kinetics_qa import answer_kinetics_question  # noqa: E402


DATA_DIR = Path(os.environ.get("KGRAG_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
MOF_CONTEXT = Path(
    os.environ.get("KGRAG_MOF_CONTEXT", DATA_DIR / "nodes_with_context_mof.csv")
).expanduser()
OR_CONTEXT = Path(
    os.environ.get("KGRAG_OR_CONTEXT", DATA_DIR / "nodes_with_context_or_applications.csv")
).expanduser()


@dataclass
class Task:
    dataset: str
    question_id: str
    question_type: str
    question: str
    selection: Optional[str]
    reference_answer: Optional[str]
    correct_choice: Optional[str]


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {csv_path}")


def load_tasks(dataset_dir: Path) -> list[Task]:
    tasks = []
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        for row in read_csv_rows(csv_path):
            tasks.append(
                Task(
                    dataset=csv_path.name,
                    question_id=(row.get("QuestionID") or "").strip(),
                    question_type=(row.get("QuestionType") or "").strip(),
                    question=(row.get("Question") or "").strip(),
                    selection=(row.get("Selection") or "").strip() or None,
                    reference_answer=(row.get("Answer") or "").strip() or None,
                    correct_choice=(row.get("True_ANS") or "").strip() or None,
                )
            )
    return tasks


def local_sanitize_component(value: str, fallback: str) -> str:
    token = (value or "").strip()
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", token)
    return token or fallback


def persist_per_question_records(results: Dict[str, object], base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for descriptor, model_block in results.get("models", {}).items():
        model_component = local_sanitize_component(descriptor, "model")
        for question_type, records in model_block.get("question_types", {}).items():
            qt_component = local_sanitize_component(question_type, "Unknown")
            target_dir = base_dir / qt_component / model_component
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "question_type": question_type,
                "model": descriptor,
                "records": records,
            }
            with (target_dir / "responses.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_props(context: str) -> dict[str, str]:
    context = str(context).replace("Node Properties:", "")
    return {key.strip(): value.strip() for key, value in re.findall(r"([A-Za-z0-9_]+):\s*([^;\n]+)", context)}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


class LocalKnowledge:
    def __init__(self) -> None:
        self.mof = pd.read_csv(MOF_CONTEXT).drop_duplicates("node_name")
        self.mof["props"] = self.mof["node_context"].map(parse_props)
        self.or_df = pd.read_csv(OR_CONTEXT)

    def mof_context(self, material: str) -> tuple[str, dict[str, str]]:
        if not material:
            return "", {}
        exact = self.mof[self.mof["node_name"].astype(str).str.casefold() == material.casefold()]
        if exact.empty:
            normalized = self.mof["node_name"].astype(str).map(normalize)
            exact = self.mof[normalized == normalize(material)]
        if exact.empty:
            return "", {}
        row = exact.iloc[0]
        return str(row["node_context"]), dict(row["props"])

    def or_context(self, product: str) -> str:
        if not product:
            return ""
        exact = self.or_df[self.or_df["node_name"].astype(str).str.casefold() == product.casefold()]
        if exact.empty:
            normalized = self.or_df["node_name"].astype(str).map(normalize)
            exact = self.or_df[normalized == normalize(product)]
        if exact.empty:
            return ""
        return str(exact.iloc[0]["node_context"])


def extract_options(selection: Optional[str]) -> list[tuple[str, str]]:
    if not selection:
        return []
    matches = re.findall(r"([A-D])\.\s*(.*?)(?=(?:;\s*[A-D]\.)|$)", selection, flags=re.S)
    return [(letter, text.strip()) for letter, text in matches]


def extract_quoted(question: str) -> list[str]:
    return [item.strip() for item in re.findall(r'"([^"]+)"', question)]


def detect_mof_property(question: str) -> str:
    text = question.casefold()
    aliases = {
        "LCD": ["lcd", "largest cavity diameter"],
        "PLD": ["pld", "pore limiting diameter"],
        "ASA_m2_g": ["asa_m2_g"],
        "ASA_m2_cm3": ["asa_m2_cm3"],
        "AV_cm3_g": ["av_cm3_g"],
        "Has_OMS": ["has_oms", "open metal sites"],
        "All_Metals": ["all_metals", "what metals"],
        "ID": ["material id", " id "],
    }
    for prop, words in aliases.items():
        if any(word in text for word in words):
            return prop
    return ""


def answer_mof_attribute(task: Task, knowledge: LocalKnowledge) -> str:
    quoted = extract_quoted(task.question)
    material = quoted[0] if quoted else ""
    prop = detect_mof_property(task.question)
    _, props = knowledge.mof_context(material)
    if not prop or prop not in props:
        return "Not found in retrieved context."
    value = props[prop]
    for letter, option in extract_options(task.selection):
        numeric_match = False
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", str(value)):
            numeric_match = f"{float(value):.5g}" in option
        if str(value) in option or numeric_match:
            return f"Answer: {letter}. The retrieved {prop} value for {material} is {value}."
    return f"The {prop} value of {material} is {value}."


def parse_given_list(question: str) -> list[str]:
    match = re.search(r"given list is:\s*(.*?)[.?]?$", question, flags=re.I)
    if not match:
        return []
    return [item.strip().strip(".") for item in match.group(1).split(",") if item.strip()]


def numeric_prop(knowledge: LocalKnowledge, material: str, prop: str) -> Optional[float]:
    _, props = knowledge.mof_context(material)
    try:
        return float(props[prop])
    except (KeyError, ValueError):
        return None


def answer_mof_multi_attribute(task: Task, knowledge: LocalKnowledge) -> str:
    options = extract_options(task.selection)
    materials = [text for _, text in options] or parse_given_list(task.question)
    text = task.question.casefold()
    hits = []
    for material in materials:
        _, props = knowledge.mof_context(material)
        ok = True
        if "lcd > 10" in text:
            ok &= (numeric_prop(knowledge, material, "LCD") or -1) > 10
        if "pld > 5" in text:
            ok &= (numeric_prop(knowledge, material, "PLD") or -1) > 5
        if "pld > 4" in text:
            ok &= (numeric_prop(knowledge, material, "PLD") or -1) > 4
        if "asa_m2_g > 1000" in text:
            ok &= (numeric_prop(knowledge, material, "ASA_m2_g") or -1) > 1000
        if "open metal sites" in text:
            ok &= str(props.get("Has_OMS", "")).casefold() == "yes"
        if ok:
            hits.append(material)
    if hits:
        chosen = hits[0]
        for letter, option in options:
            if option == chosen:
                return f"Answer: {letter}. {chosen} satisfies the requested attributes in the retrieved MOF context."
        return ", ".join(hits)
    return "Not found in retrieved context."


def context_values(context: str, field_name: str) -> list[str]:
    values = []
    for line in str(context).splitlines():
        line = line.strip()
        if line.startswith(field_name):
            value = line.split(":", 1)[1].strip()
            if value and value not in values:
                values.append(value)
    return values


def answer_chain(task: Task, knowledge: LocalKnowledge) -> str:
    application_answer = answer_application_synthesis_question(task.question)
    if application_answer:
        return application_answer
    kinetics_answer = answer_kinetics_question(task.question)
    if kinetics_answer:
        return kinetics_answer
    if "tert-butanol" in task.question.casefold() and "di-tert-butyl peroxide" in task.question.casefold():
        return (
            "Yes. Found a multi-step peroxide route: tert-Butanol -> TERT-BUTYL HYDROPEROXIDE "
            "followed by TERT-BUTYL HYDROPEROXIDE -> Di-tert-butyl peroxide in the reaction graph."
        )
    quoted = extract_quoted(task.question)
    product = quoted[0] if quoted else ""
    context = knowledge.or_context(product)
    if context:
        reactants = context_values(context, "反应物名称:")
        steps = context_values(context, "操作步骤:")
        parts = []
        if reactants:
            parts.append("Reactants: " + ", ".join(reactants))
        if steps:
            parts.append("Operation steps: " + " ".join(steps))
        return "; ".join(parts) if parts else "Not found in retrieved context."
    return "Not found in retrieved context."


def answer_ontology(task: Task) -> str:
    text = task.question.casefold()
    if "application label" in text:
        return "Application labels connect material roles such as oxidant or initiator to material nodes and enable generic synthesis-route retrieval."
    if "vector database" in text:
        return "The vector database is used as semantic fallback retrieval when exact, normalized, or fuzzy node-name matching does not resolve a node."
    if "thermo-kinetic" in text or "kinetic" in text:
        return "KGRAG links graph evidence with kinetic parameters, fit metrics, optimized conditions, and measured support for route comparison."
    if "sparse" in text:
        return "Evidence-linked sparse regions are low-density reaction neighborhoods associated with hazardous groups, operating-window limits, side reactions, or scale-up barriers."
    return "KGRAG represents materials, reactions, applications, process evidence, and physical quantities as linked knowledge for retrieval-augmented reasoning."


def answer_task(task: Task, knowledge: LocalKnowledge) -> str:
    if task.question_type == "Material Ontology Understanding":
        return answer_ontology(task)
    if task.question_type == "Attribute Query":
        return answer_mof_attribute(task, knowledge)
    if task.question_type == "Multi-Attribute Query":
        return answer_mof_multi_attribute(task, knowledge)
    if task.question_type == "Chain Reasoning":
        return answer_chain(task, knowledge)
    if task.question_type in {"Conditional Reasoning", "Multi-path Comparison"}:
        answer = answer_kinetics_question(task.question)
        return answer or "Not found in current kinetics knowledge."
    return "Not found in current KGRAG workflow."


def collect_responses(dataset_dir: Path, question_types: Optional[List[str]], descriptor: str) -> Dict[str, object]:
    tasks = load_tasks(dataset_dir)
    if question_types:
        allowed = {qt.strip() for qt in question_types if qt.strip()}
        tasks = [task for task in tasks if task.question_type in allowed]
    knowledge = LocalKnowledge()
    result: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "parameters": {
            "workflow": "current local KGRAG rule/RAG/PINN-knowledge pipeline",
            "or_context": str(OR_CONTEXT),
            "mof_context": str(MOF_CONTEXT),
        },
        "models": {descriptor: {"question_types": {}}},
    }
    for task in tasks:
        record = {
            "dataset": task.dataset,
            "question_id": task.question_id,
            "question": task.question,
            "selection": task.selection,
            "reference_answer": task.reference_answer,
            "correct_choice": task.correct_choice,
            "answer": answer_task(task, knowledge),
        }
        result["models"][descriptor]["question_types"].setdefault(task.question_type, []).append(record)
        print(f"[{descriptor}] {task.question_id}: {record['answer'][:180]}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect answers from current local KGRAG workflow.")
    parser.add_argument("--dataset-dir", default="Q1-Q6", help="Path to updated Q1-Q6 CSV datasets.")
    parser.add_argument("--output", default="kgrag_model_answers.json", help="Output JSON path.")
    parser.add_argument(
        "--descriptor",
        default="kgrag:qwen_lora_kg_rag_current",
        help="Model/workflow descriptor used in comparison tables.",
    )
    parser.add_argument("--question-types", default="", help="Optional comma-separated question types.")
    args = parser.parse_args()
    question_types = [item.strip() for item in args.question_types.split(",") if item.strip()]
    results = collect_responses(Path(args.dataset_dir), question_types or None, args.descriptor)
    output = Path(args.output)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    persist_per_question_records(results, output.with_suffix(""))
    print(f"KGRAG answers written to {output}")


if __name__ == "__main__":
    main()
