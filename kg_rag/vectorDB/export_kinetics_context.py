import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("KGRAG_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
KINETICS_OUTPUTS = Path(
    os.environ.get("KGRAG_KINETICS_OUTPUTS", DATA_DIR / "private/kinetics_outputs")
).expanduser()
CONTEXT_OUTPUT = DATA_DIR / "nodes_with_context_kinetics.csv"
JSON_OUTPUT = DATA_DIR / "kinetics_route_knowledge.json"


ROUTE_LABELS = {
    "tbhp_bzcl": "TBHP-BZCL",
    "tbhp_cf3": "TBHP-CF3",
    "tbhp_wpo4": "TBHP-TBA-WPO4",
    "tbpb_benzaldehyde": "TBPB-benzaldehyde",
}


REACTION_EQUATIONS = {
    "tbhp_bzcl": [
        "r1 = k1 * [H2O2]^order_H2O2 * [TBCL]^order_TBCL",
        "k1 = exp(lnA1 - Ea1_over_R / T)",
    ],
    "tbhp_cf3": [
        "r1 = k1 * [H2O2]^order_H2O2 * [TBA]^order_TBA_main",
        "k1 = exp(lnA1 + cat_order_1 * ln(Ccat) - Ea1_over_R / T)",
        "r2 = k2 * [TBA]^order_TBA_side * [TBHP]^order_TBHP_side",
        "k2 = exp(lnA2 + cat_order_2 * ln(Ccat) - Ea2_over_R / T)",
    ],
    "tbhp_wpo4": [
        "r1 = k1 * [H2O2]^order_H2O2 * [TBA]^order_TBA_main",
        "k1 = exp(lnA1 + cat_order_1 * ln(Ccat) - Ea1_over_R / T)",
        "r2 = k2 * [TBA]^order_TBA_side * [TBHP]^order_TBHP_side",
        "k2 = exp(lnA2 + cat_order_2 * ln(Ccat) - Ea2_over_R / T)",
    ],
    "tbpb_benzaldehyde": [
        "r1 = k1 * [Benzaldehyde]^order_benzaldehyde * [TBHP]^order_TBHP",
        "k1 = exp(lnA1 + cat_order_1 * ln(Ccat) - Ea1_over_R / T)",
    ],
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_dict(prefix, data):
    lines = []
    for key, value in data.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.6g}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines


def condition_lines(title, condition):
    lines = [f"{title}:"]
    for key, value in condition.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6g}")
        elif isinstance(value, (int, str)):
            lines.append(f"{key}: {value}")
    return lines


def make_route_context(route_key, ode_summary, opt_summary):
    route_label = ROUTE_LABELS.get(route_key, route_key)
    learned = ode_summary.get("learned_params", {})
    metrics = ode_summary.get("metrics", {})
    best_model = opt_summary.get("best_model", {})
    best_measured = opt_summary.get("best_measured", {})
    nearest_measured = opt_summary.get("nearest_measured_to_model", {})
    consensus = opt_summary.get("consensus_recommendation", best_model)

    lines = [
        f"Kinetic route name: {route_label}",
        f"Route key: {route_key}",
        f"Model type: {ode_summary.get('model_type', '')}",
        f"Trainer: {ode_summary.get('trainer', '')}",
        f"PINN initialization: {ode_summary.get('initialization', '')}",
        f"Epochs: {ode_summary.get('epochs', '')}",
        f"Constrained parameters: {ode_summary.get('constrain_params', False)}",
        f"Optimization objective: {opt_summary.get('objective', '')}",
        "Reaction rate equations:",
    ]
    lines.extend(REACTION_EQUATIONS.get(route_key, []))
    lines.append("PINN learned kinetic parameters:")
    lines.extend(format_dict("", learned))
    lines.append("PINN fit quality metrics:")
    lines.extend(format_dict("", metrics))
    lines.extend(condition_lines("PINN recommended optimum condition", consensus))
    lines.extend(condition_lines("PINN combinatorial best model condition", best_model))
    lines.extend(condition_lines("Best measured experimental condition", best_measured))
    lines.extend(condition_lines("Nearest measured support condition", nearest_measured))
    lines.extend(
        [
            "Interpretation note: These kinetic parameters were learned by ODE-inverse PINN from small experimental datasets.",
            "Interpretation note: Conventional curve-fit values are reference only; PINN initial values were neutral, not initialized from curve_fit.",
            "Use case: compare synthesis paths by activation energy, predicted yield, selectivity, conversion, side-product penalty, and measured support.",
        ]
    )
    return "\n".join(lines)


def main():
    opt_summary = load_json(KINETICS_OUTPUTS / "condition_optimization_summary.json")
    records = []
    knowledge = {}

    for summary_path in sorted(KINETICS_OUTPUTS.glob("*_ode_summary.json")):
        ode_summary = load_json(summary_path)
        route_key = ode_summary["route"]
        context = make_route_context(route_key, ode_summary, opt_summary[route_key])
        node_name = f"kinetics::{ROUTE_LABELS.get(route_key, route_key)}"
        records.append({"node_name": node_name, "node_context": context})
        knowledge[route_key] = {
            "node_name": node_name,
            "route_label": ROUTE_LABELS.get(route_key, route_key),
            "ode_summary": ode_summary,
            "condition_optimization": opt_summary[route_key],
            "context": context,
        }

    CONTEXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(CONTEXT_OUTPUT, index=False)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(knowledge, file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} kinetics nodes to {CONTEXT_OUTPUT}")
    print(f"Wrote machine-readable knowledge to {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
