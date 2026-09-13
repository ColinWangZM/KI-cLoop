#!/usr/bin/env python3
"""Build a generic OR context table with material application labels.

The output keeps the standard `node_name,node_context` schema so existing OR QA
code can use it. Application labels are stored in node_context rather than
changing node_name, preserving exact product-name lookup.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("KGRAG_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
KINETICS_ROOT = Path(
    os.environ.get("KGRAG_KINETICS_ROOT", DATA_DIR / "private/kinetics")
).expanduser()
SOURCE_CONTEXTS = [
    DATA_DIR / "nodes_with_context_or.csv",
    DATA_DIR / "nodes_with_context_or_YT.csv",
]
OUTPUT_CONTEXT = DATA_DIR / "nodes_with_context_or_applications.csv"
OUTPUT_INDEX = DATA_DIR / "material_application_index.json"


APPLICATION_RULES = {
    "oxidant": [
        "hydrogen peroxide",
        "h2o2",
        "tert-butyl hydroperoxide",
        "tbhp",
        "thbp",
        "peroxide",
        "hydroperoxide",
        "m-chloroperbenzoic acid",
        "mcpba",
        "sodium hypochlorite",
        "potassium permanganate",
    ],
    "initiator": [
        "tert-butyl hydroperoxide",
        "tbhp",
        "thbp",
        "di-tert-butyl peroxide",
        "dtbp",
        "tert-butyl perbenzoate",
        "tert-butyl peroxybenzoate",
        "tbpb",
        "benzoyl peroxide",
        "aibn",
        "azobisisobutyronitrile",
        "peroxide",
        "hydroperoxide",
    ],
    "solvent": [
        "methanol",
        "ethanol",
        "acetonitrile",
        "dichloromethane",
        "chloroform",
        "benzene",
        "toluene",
        "water",
        "pyridine",
    ],
    "acid_catalyst": [
        "sulfuric acid",
        "hydrochloric acid",
        "trifluoroacetic acid",
        "p-toluenesulfonic acid",
        "phosphotungstic acid",
        "wpo4",
    ],
    "base": [
        "sodium hydroxide",
        "potassium hydroxide",
        "triethylamine",
        "piperidine",
        "pyridine",
        "sodium carbonate",
        "potassium carbonate",
    ],
    "acylating_agent": [
        "benzoyl chloride",
        "acid chloride",
        "chloroformate",
        "anhydride",
    ],
    "substrate": [
        "benzaldehyde",
        "tert-butanol",
        "tert butanol",
        "tba",
        "alcohol",
        "aldehyde",
    ],
}


CURATED_MATERIALS = {
    "hydrogen peroxide": {
        "aliases": ["H2O2", "hydrogen peroxide"],
        "applications": ["oxidant"],
    },
    "TERT-BUTYL HYDROPEROXIDE": {
        "aliases": [
            "TBHP",
            "THBP",
            "tert-butyl hydroperoxide",
            "tert butyl hydroperoxide",
            "TERT-BUTYL HYDROPEROXIDE",
        ],
        "applications": ["oxidant", "initiator", "initiator precursor"],
    },
    "Di-tert-butyl peroxide": {
        "aliases": ["DTBP", "di-tert-butyl peroxide", "di tert butyl peroxide"],
        "applications": ["initiator"],
    },
    "tert-butyl perbenzoate": {
        "aliases": [
            "TBPB",
            "tert-butyl perbenzoate",
            "tert butyl perbenzoate",
            "tert-butyl peroxybenzoate",
        ],
        "applications": ["initiator"],
    },
    "benzoyl peroxide": {
        "aliases": ["benzoyl peroxide", "BPO"],
        "applications": ["initiator"],
    },
    "azobisisobutyronitrile": {
        "aliases": ["AIBN", "azobisisobutyronitrile"],
        "applications": ["initiator"],
    },
    "benzaldehyde": {
        "aliases": ["benzaldehyde"],
        "applications": ["substrate"],
    },
    "tert-Butanol": {
        "aliases": ["tert-Butanol", "tert-butanol", "TBA"],
        "applications": ["substrate", "solvent"],
    },
    "benzoyl chloride": {
        "aliases": ["benzoyl chloride", "BZCL", "TBCL"],
        "applications": ["acylating_agent", "substrate"],
    },
}


KINETIC_ROUTE_NODES = [
    {
        "node_name": "TERT-BUTYL HYDROPEROXIDE",
        "route_label": "TBHP-BZCL",
        "reactants": ["hydrogen peroxide", "benzoyl chloride/TBCL"],
        "product": "TERT-BUTYL HYDROPEROXIDE",
        "applications": ["oxidant", "initiator", "initiator precursor"],
        "source": KINETICS_ROOT / "TBHP-BZCL",
        "summary": "Kinetics-derived experimental route: H2O2 + TBCL/BZCL -> TBHP.",
    },
    {
        "node_name": "TERT-BUTYL HYDROPEROXIDE",
        "route_label": "TBHP-CF3",
        "reactants": ["hydrogen peroxide", "tert-Butanol", "CF3 catalyst"],
        "product": "TERT-BUTYL HYDROPEROXIDE",
        "applications": ["oxidant", "initiator", "initiator precursor"],
        "source": KINETICS_ROOT / "TBHP-CF3",
        "summary": "Kinetics-derived experimental route: H2O2 + TBA -> TBHP with DTBP side formation.",
    },
    {
        "node_name": "TERT-BUTYL HYDROPEROXIDE",
        "route_label": "TBHP-TBA-WPO4",
        "reactants": ["hydrogen peroxide", "tert-Butanol", "WPO4 catalyst"],
        "product": "TERT-BUTYL HYDROPEROXIDE",
        "applications": ["oxidant", "initiator", "initiator precursor"],
        "source": KINETICS_ROOT / "TBHP-TBA-WPO4",
        "summary": "Kinetics-derived experimental route: H2O2 + TBA -> TBHP with WPO4 catalyst and DTBP side formation.",
    },
    {
        "node_name": "tert-butyl perbenzoate",
        "route_label": "TBPB-benzaldehyde",
        "reactants": ["benzaldehyde", "TERT-BUTYL HYDROPEROXIDE"],
        "product": "tert-butyl perbenzoate",
        "applications": ["initiator"],
        "source": KINETICS_ROOT / "TBPB-苯甲醛",
        "summary": "Kinetics-derived experimental route: benzaldehyde + TBHP -> TBPB + TBA.",
    },
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def load_source_contexts() -> pd.DataFrame:
    frames = []
    for path in SOURCE_CONTEXTS:
        if path.exists():
            frame = pd.read_csv(path, usecols=["node_name", "node_context"])
            frame["source_file"] = str(path)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"Missing source OR CSVs: {SOURCE_CONTEXTS}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["node_name", "node_context"]
    )


def context_values(context: str, label: str) -> list[str]:
    values = []
    for line in str(context).splitlines():
        if line.startswith(label):
            values.append(line.split(":", 1)[1].strip())
    return values


def match_applications(text: str) -> set[str]:
    text_norm = normalize(text)
    labels = set()
    for application, terms in APPLICATION_RULES.items():
        if any(normalize(term) in text_norm for term in terms):
            labels.add(application)
    return labels


def curated_labels_for_name(name: str) -> tuple[set[str], list[str]]:
    name_norm = normalize(name)
    labels = set()
    aliases = []
    for material, data in CURATED_MATERIALS.items():
        all_names = [material] + data["aliases"]
        if any(normalize(alias) == name_norm for alias in all_names):
            labels.update(data["applications"])
            aliases.extend(data["aliases"])
    return labels, sorted(set(aliases))


def append_application_context(row: pd.Series) -> tuple[str, dict]:
    node_name = str(row["node_name"])
    context = str(row["node_context"])
    labels = match_applications(node_name)
    curated_labels, aliases = curated_labels_for_name(node_name)
    labels.update(curated_labels)
    if not labels:
        return context, {}

    reactants = context_values(context, "反应物名称:")
    lines = [
        context,
        "应用标签: " + ", ".join(sorted(labels)),
        "应用标签来源: generic_rule_or_curated_material_alias",
    ]
    if aliases:
        lines.append("别名: " + ", ".join(aliases))
    if reactants:
        lines.append("合成路径反应物: " + ", ".join(reactants))
    lines.append(f"合成路径产物: {node_name}")
    record = {
        "material": node_name,
        "applications": sorted(labels),
        "aliases": aliases,
        "reactants": reactants,
        "product": node_name,
        "source_file": row.get("source_file", ""),
        "route_source": "organic_reaction_context",
        "node_context": "\n".join(lines),
    }
    return "\n".join(lines), record


def kinetic_route_context(route: dict) -> str:
    lines = [
        route["summary"],
        "应用标签: " + ", ".join(route["applications"]),
        "应用标签来源: kinetics_experimental_route",
        "别名: " + ", ".join(CURATED_MATERIALS.get(route["node_name"], {}).get("aliases", [])),
        "合成路径名称: " + route["route_label"],
        "合成路径反应物: " + ", ".join(route["reactants"]),
        "合成路径产物: " + route["product"],
        f"实验数据路径: {route['source']}",
    ]
    return "\n".join(lines)


def build_application_index(records: list[dict]) -> dict:
    index = {}
    for record in records:
        material = record["material"]
        entry = index.setdefault(
            material,
            {
                "material": material,
                "applications": set(),
                "aliases": set(),
                "synthesis_routes": [],
            },
        )
        entry["applications"].update(record.get("applications", []))
        entry["aliases"].update(record.get("aliases", []))
        entry["synthesis_routes"].append(
            {
                "reactants": record.get("reactants", []),
                "product": record.get("product", material),
                "route_label": record.get("route_label", material),
                "route_source": record.get("route_source", ""),
                "source_file": str(record.get("source_file", "")),
            }
        )
    return {
        material: {
            **entry,
            "applications": sorted(entry["applications"]),
            "aliases": sorted(entry["aliases"]),
        }
        for material, entry in sorted(index.items())
    }


def main() -> None:
    source = load_source_contexts()
    augmented_rows = []
    index_records = []

    for _, row in source.iterrows():
        context, record = append_application_context(row)
        augmented_rows.append({"node_name": row["node_name"], "node_context": context})
        if record:
            index_records.append(record)

    for route in KINETIC_ROUTE_NODES:
        augmented_rows.append(
            {"node_name": route["node_name"], "node_context": kinetic_route_context(route)}
        )
        index_records.append(
            {
                "material": route["node_name"],
                "applications": route["applications"],
                "aliases": CURATED_MATERIALS.get(route["node_name"], {}).get("aliases", []),
                "reactants": route["reactants"],
                "product": route["product"],
                "route_label": route["route_label"],
                "route_source": "kinetics_experimental_route",
                "source_file": str(route["source"]),
            }
        )

    augmented = pd.DataFrame(augmented_rows)
    augmented.to_csv(OUTPUT_CONTEXT, index=False)
    index = build_application_index(index_records)
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(augmented)} OR application contexts to {OUTPUT_CONTEXT}")
    print(f"Wrote {len(index)} application-tagged materials to {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
