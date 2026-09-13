import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("KGRAG_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
KINETICS_KNOWLEDGE_PATH = Path(
    os.environ.get("KGRAG_KINETICS_KNOWLEDGE", DATA_DIR / "kinetics_route_knowledge.json")
).expanduser()


ROUTE_ALIASES = {
    "tbhp_bzcl": ["tbhp-bzcl", "bzcl", "tbcl"],
    "tbhp_cf3": ["tbhp-cf3", "cf3"],
    "tbhp_wpo4": ["tbhp-tba-wpo4", "wpo4", "tba-wpo4"],
    "tbpb_benzaldehyde": ["tbpb-benzaldehyde", "tbpb", "benzaldehyde"],
}


def load_kinetics_knowledge(path=KINETICS_KNOWLEDGE_PATH):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def detect_routes(question, knowledge):
    question_norm = normalize(question)
    detected = []
    for route_key, route_data in knowledge.items():
        labels = [route_key, route_data.get("route_label", "")]
        labels.extend(ROUTE_ALIASES.get(route_key, []))
        if any(normalize(label) in question_norm for label in labels):
            detected.append(route_key)
    return detected


def route_title(route_key, knowledge):
    return knowledge[route_key].get("route_label", route_key)


def learned_params(route_key, knowledge):
    return knowledge[route_key]["ode_summary"].get("learned_params", {})


def metrics(route_key, knowledge):
    return knowledge[route_key]["ode_summary"].get("metrics", {})


def recommendation(route_key, knowledge):
    opt = knowledge[route_key]["condition_optimization"]
    return opt.get("consensus_recommendation") or opt.get("best_model", {})


def best_measured(route_key, knowledge):
    return knowledge[route_key]["condition_optimization"].get("best_measured", {})


def objective(route_key, knowledge):
    return knowledge[route_key]["condition_optimization"].get("objective", "")


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_dict(data, keys=None):
    items = data.items() if keys is None else [(key, data[key]) for key in keys if key in data]
    return "\n".join(f"- {key}: {format_value(value)}" for key, value in items)


def compact_condition(data, prefer_initial=False):
    base_condition_keys = [
        "T",
        "time",
        "H2O2",
        "TBA",
        "TBCL",
        "Benzaldehyde",
        "Catalyst",
        "TBHP",
        "DTBP",
        "TBPB",
    ]
    condition_keys = []
    for key in base_condition_keys:
        if prefer_initial and f"{key}_0" in data:
            condition_keys.append(f"{key}_0")
        else:
            condition_keys.append(key)
    performance_keys = ["score", "yield", "selectivity", "conversion", "impurity", "basis"]
    lines = []
    condition = format_dict(data, condition_keys)
    performance = format_dict(data, performance_keys)
    if condition:
        lines.append("Condition:")
        lines.append(condition)
    if performance:
        lines.append("Predicted performance:")
        lines.append(performance)
    return "\n".join(lines)


def answer_route_parameters(route_key, knowledge):
    return (
        f"Kinetic parameters for {route_title(route_key, knowledge)}:\n"
        + format_dict(learned_params(route_key, knowledge))
        + "\nFit metrics:\n"
        + format_dict(metrics(route_key, knowledge))
    )


def answer_route_recommendation(route_key, knowledge):
    return (
        f"Recommended condition for {route_title(route_key, knowledge)} based on PINN kinetics:\n"
        + compact_condition(recommendation(route_key, knowledge))
        + "\nOptimization objective:\n"
        + objective(route_key, knowledge)
        + "\nBest measured support:\n"
        + compact_condition(best_measured(route_key, knowledge), prefer_initial=True)
    )


def answer_compare_routes(route_keys, knowledge):
    rows = []
    for route_key in route_keys:
        rec = recommendation(route_key, knowledge)
        params = learned_params(route_key, knowledge)
        metric = metrics(route_key, knowledge)
        rows.append(
            {
                "route": route_title(route_key, knowledge),
                "score": rec.get("score"),
                "yield": rec.get("yield"),
                "selectivity": rec.get("selectivity"),
                "conversion": rec.get("conversion"),
                "Ea1_kJ_mol": params.get("Ea1_kJ_mol"),
                "Ea1_over_R": params.get("Ea1_over_R"),
                "R2_product": next(
                    (value for key, value in metric.items() if key.startswith("R2_TBHP") or key.startswith("R2_TBPB")),
                    None,
                ),
            }
        )

    ranked = sorted(rows, key=lambda row: (row["score"] is None, -(row["score"] or -1)))
    lines = ["Route comparison based on PINN kinetic optimization score:"]
    for row in ranked:
        lines.append(
            "- "
            + row["route"]
            + f": score={format_value(row['score'])}, yield={format_value(row['yield'])}, "
            + f"selectivity={format_value(row['selectivity'])}, conversion={format_value(row['conversion'])}, "
            + f"Ea1_kJ_mol={format_value(row['Ea1_kJ_mol'])}, R2_product={format_value(row['R2_product'])}"
        )
    lines.append(
        "Selection rule: prefer higher optimization score and yield/selectivity, then check activation energy and measured support."
    )
    return "\n".join(lines)


def answer_kinetics_question(question, knowledge=None):
    knowledge = knowledge or load_kinetics_knowledge()
    question_lower = str(question).casefold()
    route_keys = detect_routes(question, knowledge)

    kinetics_scope_terms = [
        "kinetic",
        "parameter",
        "activation",
        "ea",
        "lna",
        "order",
        "peroxide",
        "yield",
        "selectivity",
        "score",
        "synthesis",
        "condition",
        "recommend",
        "optimum",
        "optimal",
        "动力学",
        "参数",
        "活化能",
        "产率",
        "收率",
        "选择性",
        "条件",
        "推荐",
        "过氧化物",
        "路径",
        "路线",
    ]
    route_selection_terms = [
        "choose",
        "select",
        "compare",
        "which route",
        "which path",
        "哪条",
        "哪个",
        "选择",
        "对比",
        "比较",
        "更好",
        "最佳",
    ]
    if not route_keys and any(term in question_lower for term in route_selection_terms) and any(
        term in question_lower for term in kinetics_scope_terms
    ):
        route_keys = list(knowledge.keys())

    if not route_keys:
        return ""

    if any(term in question_lower for term in route_selection_terms + ["best"]):
        return answer_compare_routes(route_keys, knowledge)

    route_key = route_keys[0]
    if any(term in question_lower for term in ["condition", "recommend", "optimum", "optimal", "yield", "selectivity"]):
        return answer_route_recommendation(route_key, knowledge)

    if any(term in question_lower for term in ["parameter", "kinetic", "activation", "ea", "lna", "order"]):
        return answer_route_parameters(route_key, knowledge)

    return answer_route_parameters(route_key, knowledge)


if __name__ == "__main__":
    examples = [
        "What are the kinetic parameters for TBHP-BZCL?",
        "What condition is recommended for TBHP-TBA-WPO4?",
        "Compare all peroxide synthesis routes by activation energy, yield and selectivity.",
    ]
    for example in examples:
        print("\nQuestion:", example)
        print(answer_kinetics_question(example))
