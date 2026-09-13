#!/usr/bin/env python3
"""Build Q1-Q6 evaluation datasets aligned with the current KGRAG code.

The question-type names are intentionally kept unchanged so existing
collection and judging scripts continue to work.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("KGRAG_DATA_DIR", ROOT / "data")).expanduser()
DATASET_DIR = EVAL_ROOT / "Q1-Q6"
MOF_CONTEXT = Path(
    os.environ.get("KGRAG_MOF_CONTEXT", DATA_DIR / "nodes_with_context_mof.csv")
).expanduser()
OR_CONTEXT = Path(
    os.environ.get("KGRAG_OR_CONTEXT", DATA_DIR / "nodes_with_context_or_applications.csv")
).expanduser()
KINETICS_KNOWLEDGE = Path(
    os.environ.get("KGRAG_KINETICS_KNOWLEDGE", DATA_DIR / "kinetics_route_knowledge.json")
).expanduser()

RNG = random.Random(42)


def parse_props(context: str) -> dict[str, str]:
    context = str(context).replace("Node Properties:", "")
    return {key.strip(): value.strip() for key, value in re.findall(r"([A-Za-z0-9_]+):\s*([^;\n]+)", context)}


def unique_mof_rows() -> pd.DataFrame:
    df = pd.read_csv(MOF_CONTEXT).drop_duplicates("node_name").copy()
    parsed = df["node_context"].map(parse_props)
    for key in ["LCD", "PLD", "ASA_m2_g", "ASA_m2_cm3", "AV_cm3_g", "Has_OMS", "All_Metals", "ID"]:
        df[key] = parsed.map(lambda props, k=key: props.get(k))
    for key in ["LCD", "PLD", "ASA_m2_g", "ASA_m2_cm3", "AV_cm3_g"]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    return df.dropna(subset=["LCD", "PLD"]).reset_index(drop=True)


def option_string(options: list[str]) -> tuple[str, str]:
    letters = ["A", "B", "C", "D"]
    return "; ".join(f"{letter}. {option}" for letter, option in zip(letters, options, strict=False)), letters[0]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str], encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_q1() -> None:
    rows = [
        {
            "QuestionID": "Q1-01",
            "QuestionType": "Material Ontology Understanding",
            "Question": "What does a node represent in the updated KGRAG material and reaction graph?",
            "Answer": "A node represents a material, reaction product, route, or physical quantity entry with structured context such as properties, applications, reactants, operations, or kinetic parameters.",
        },
        {
            "QuestionID": "Q1-02",
            "QuestionType": "Material Ontology Understanding",
            "Question": "Why are application labels useful for finding oxidant or initiator synthesis routes?",
            "Answer": "Application labels connect material roles such as oxidant or initiator to material nodes, enabling generic retrieval of candidate synthesis routes before manual experimental screening.",
        },
        {
            "QuestionID": "Q1-03",
            "QuestionType": "Material Ontology Understanding",
            "Question": "What is the role of the vector database in the updated KGRAG workflow?",
            "Answer": "The vector database provides semantic fallback retrieval when exact, normalized, or fuzzy node-name matching cannot identify the correct graph node.",
        },
        {
            "QuestionID": "Q1-04",
            "QuestionType": "Material Ontology Understanding",
            "Question": "Why does KGRAG combine graph evidence with thermo-kinetic quantities?",
            "Answer": "Graph evidence identifies reaction species and routes, while thermo-kinetic quantities such as activation energy, reaction orders, yield, and selectivity support condition recommendation and route comparison.",
        },
        {
            "QuestionID": "Q1-05",
            "QuestionType": "Material Ontology Understanding",
            "Question": "What is meant by evidence-linked sparse reaction regions in the reaction-space map?",
            "Answer": "They are low-density reaction neighborhoods that also carry chemical evidence such as hazardous groups, demanding operating windows, mechanistic uncertainty, or scale-up constraints.",
        },
        {
            "QuestionID": "Q1-06",
            "QuestionType": "Material Ontology Understanding",
            "Question": "Why should global distances in a two-dimensional reaction-space map not be over-interpreted?",
            "Answer": "Dimensionality reduction mainly preserves local neighborhoods, so nearby points are informative but long-range two-dimensional distances are not strict quantitative chemical distances.",
        },
        {
            "QuestionID": "Q1-07",
            "QuestionType": "Material Ontology Understanding",
            "Question": "What information is stored in the updated organic reaction node context?",
            "Answer": "Organic reaction node context stores reactant names and identifiers, product identifiers, solvent information when available, operation steps, and application tags for relevant materials.",
        },
        {
            "QuestionID": "Q1-08",
            "QuestionType": "Material Ontology Understanding",
            "Question": "How does PINN-derived kinetic knowledge extend the original graph RAG system?",
            "Answer": "PINN-derived knowledge adds learned kinetic parameters, fit metrics, optimized conditions, and measured support, allowing the QA system to answer dynamic route-selection questions.",
        },
    ]
    write_csv(DATASET_DIR / "Q1_Ontology.csv", rows, ["QuestionID", "QuestionType", "Question", "Answer"])


def build_q2(mof: pd.DataFrame) -> None:
    property_specs = [
        ("LCD", "largest cavity diameter (LCD)", "Å"),
        ("PLD", "pore limiting diameter (PLD)", "Å"),
        ("ASA_m2_g", "accessible surface area ASA_m2_g", "m²/g"),
        ("ASA_m2_cm3", "accessible surface area ASA_m2_cm3", "m²/cm³"),
        ("AV_cm3_g", "accessible volume AV_cm3_g", "cm³/g"),
    ]
    candidates = mof.dropna(subset=[spec[0] for spec in property_specs]).sample(
        min(40, len(mof)), random_state=42
    )
    rows = []
    qid = 1
    for _, row in candidates.iterrows():
        prop, label, unit = property_specs[(qid - 1) % len(property_specs)]
        value = float(row[prop])
        distractors = [value * 0.93, value * 1.04, value + 0.137]
        options = [f"The {label} of {row.node_name} is {value:.5g} {unit}."]
        options += [f"The {label} of {row.node_name} is {v:.5g} {unit}." for v in distractors]
        RNG.shuffle(options)
        true_idx = options.index(f"The {label} of {row.node_name} is {value:.5g} {unit}.")
        letters = ["A", "B", "C", "D"]
        rows.append(
            {
                "QuestionID": f"Q2-{qid:03d}",
                "QuestionType": "Attribute Query",
                "Question": f'What is the {label} of "{row.node_name}"?',
                "Answer": f"The {label} of {row.node_name} is {value:.5g} {unit}.",
                "Selection": "; ".join(f"{letter}. {option}" for letter, option in zip(letters, options, strict=False)),
                "True_ANS": letters[true_idx],
            }
        )
        qid += 1
    write_csv(
        DATASET_DIR / "Q2_MC_questions_fixed2.csv",
        rows,
        ["QuestionID", "QuestionType", "Question", "Answer", "Selection", "True_ANS"],
    )


def build_q3(mof: pd.DataFrame) -> None:
    rows = []
    pools = mof.dropna(subset=["LCD", "PLD", "Has_OMS"]).copy()
    oms_hits = pools[pools["Has_OMS"].astype(str).str.casefold().eq("yes")]
    no_oms = pools[~pools["Has_OMS"].astype(str).str.casefold().eq("yes")]
    qid = 1
    for hit_df, question_template in [
        (
            pools[(pools["LCD"] > 10) & (pools["PLD"] > 5)],
            'Which material in the given list has both LCD > 10 Å and PLD > 5 Å? Given list is: {options}.',
        ),
        (
            oms_hits[oms_hits["LCD"] > 10],
            'Which material in the given list has open metal sites and LCD > 10 Å? Given list is: {options}.',
        ),
        (
            pools[(pools["ASA_m2_g"] > 1000) & (pools["PLD"] > 4)],
            'Which material in the given list has ASA_m2_g > 1000 m²/g and PLD > 4 Å? Given list is: {options}.',
        ),
    ]:
        for _, hit in hit_df.head(12).iterrows():
            distractors = no_oms.sample(3, random_state=100 + qid)["node_name"].tolist()
            options = [hit.node_name] + distractors
            RNG.shuffle(options)
            letters = ["A", "B", "C", "D"]
            rows.append(
                {
                    "QuestionID": f"Q3-{qid:03d}",
                    "QuestionType": "Multi-Attribute Query",
                    "Question": question_template.format(options=", ".join(options)),
                    "Answer": str(hit.node_name),
                    "Selection": "; ".join(f"{letter}. {name}" for letter, name in zip(letters, options, strict=False)),
                    "True_ANS": letters[options.index(hit.node_name)],
                }
            )
            qid += 1
    write_csv(
        DATASET_DIR / "Q3_MC_questions_fixed.csv",
        rows,
        ["QuestionID", "QuestionType", "Question", "Answer", "Selection", "True_ANS"],
    )


def context_values(context: str, field_name: str) -> list[str]:
    values = []
    for line in str(context).splitlines():
        line = line.strip()
        if line.startswith(field_name):
            value = line.split(":", 1)[1].strip()
            if value and value not in values:
                values.append(value)
    return values


def operation_steps(context: str) -> list[str]:
    return context_values(context, "操作步骤:")


def build_q4() -> None:
    or_df = pd.read_csv(OR_CONTEXT)
    rows = []
    examples = or_df.head(8)
    for idx, row in enumerate(examples.itertuples(index=False), start=1):
        reactants = context_values(row.node_context, "反应物名称:")
        steps = operation_steps(row.node_context)
        answer = []
        if reactants:
            answer.append("Reactants: " + ", ".join(reactants))
        if steps:
            answer.append("Operation steps: " + " ".join(steps))
        rows.append(
            {
                "QuestionID": f"Q4-{idx:03d}",
                "QuestionType": "Chain Reasoning",
                "Question": f'What is the synthesis route of "{row.node_name}"?',
                "Answer": "; ".join(answer) if answer else "Not found in retrieved context.",
            }
        )
    rows.extend(
        [
            {
                "QuestionID": "Q4-009",
                "QuestionType": "Chain Reasoning",
                "Question": 'Can "Di-tert-butyl peroxide" be generated from "tert-Butanol" through a multi-step path?',
                "Answer": "Yes. tert-Butanol can be converted to TERT-BUTYL HYDROPEROXIDE, and TERT-BUTYL HYDROPEROXIDE can then form Di-tert-butyl peroxide in the current reaction graph.",
            },
            {
                "QuestionID": "Q4-010",
                "QuestionType": "Chain Reasoning",
                "Question": "Find synthesis routes for materials used as oxidants or initiators.",
                "Answer": "The updated application-guided OR QA searches application labels such as oxidant and initiator, then returns synthesis-route candidates for matched materials including TBHP and TBPB-related peroxide compounds.",
            },
        ]
    )
    write_csv(DATASET_DIR / "Q4_chain_reasoning.csv", rows, ["QuestionID", "QuestionType", "Question", "Answer"])


def build_q5_q6() -> None:
    knowledge = json.loads(KINETICS_KNOWLEDGE.read_text(encoding="utf-8"))
    q5 = []
    qid = 1
    for route_key, route in knowledge.items():
        label = route["route_label"]
        rec = route["condition_optimization"].get("consensus_recommendation") or route["condition_optimization"]["best_model"]
        params = route["ode_summary"]["learned_params"]
        q5.append(
            {
                "QuestionID": f"Q5_{qid:03d}",
                "QuestionType": "Conditional Reasoning",
                "Question": f"What are the kinetic parameters for {label}?",
                "Answer": f"{label} has Ea1_kJ_mol={params.get('Ea1_kJ_mol')}, Ea1_over_R={params.get('Ea1_over_R')}, lnA1={params.get('lnA1')}, with route-specific reaction orders reported in the current PINN knowledge file.",
            }
        )
        qid += 1
        q5.append(
            {
                "QuestionID": f"Q5_{qid:03d}",
                "QuestionType": "Conditional Reasoning",
                "Question": f"What condition is recommended for {label}?",
                "Answer": f"The PINN combinatorial recommendation for {label} is {rec}; it should be compared with best measured support before experiment.",
            }
        )
        qid += 1
    write_csv(DATASET_DIR / "Q5_conditional_reasoning.csv", q5, ["QuestionID", "QuestionType", "Question", "Answer"])

    q6_questions = [
        "Which peroxide route should be chosen based on activation energy, yield and selectivity?",
        "Among the TBHP synthesis routes, which route has the best PINN optimization score?",
        "Compare TBHP-BZCL and TBHP-TBA-WPO4 for closed-loop experimental recommendation.",
        "Compare TBHP-CF3 and TBHP-TBA-WPO4 considering yield, selectivity and activation energy.",
        "Which route is preferable when measured support and PINN-predicted improvement are both considered?",
        "Which route should be prioritized for TBPB synthesis under the current kinetics knowledge?",
    ]
    q6 = []
    for idx, question in enumerate(q6_questions, start=1):
        q6.append(
            {
                "QuestionID": f"Q6_{idx:03d}",
                "QuestionType": "Multi-path Comparison",
                "Question": question,
                "Answer": "Compare routes using the current PINN optimization score, predicted yield, selectivity, activation energy, and best measured support; prefer the route with stronger combined score and experimental support rather than activation energy alone.",
            }
        )
    write_csv(DATASET_DIR / "Q6_Multi-path_Comparison.csv", q6, ["QuestionID", "QuestionType", "Question", "Answer"], encoding="utf-8-sig")


def main() -> None:
    mof = unique_mof_rows()
    build_q1()
    build_q2(mof)
    build_q3(mof)
    build_q4()
    build_q5_q6()
    print(f"Updated Q1-Q6 datasets written to {DATASET_DIR}")


if __name__ == "__main__":
    main()
