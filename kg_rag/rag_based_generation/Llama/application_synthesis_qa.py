import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("KGRAG_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
APPLICATION_INDEX_PATH = Path(
    os.environ.get("KGRAG_APPLICATION_INDEX", DATA_DIR / "material_application_index.json")
).expanduser()


APPLICATION_KEYWORDS = {
    "oxidant": ["氧化剂", "oxidant", "oxidants", "oxidation"],
    "initiator": ["引发剂", "initiator", "initiators", "initiation", "polymerization"],
    "solvent": ["溶剂", "solvent", "solvents"],
    "acid_catalyst": ["酸催化剂", "酸", "acid catalyst"],
    "base": ["碱", "base"],
    "acylating_agent": ["酰化剂", "acylating"],
    "substrate": ["底物", "substrate"],
}


def normalize(text):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text).casefold())


def keyword_in_question(keyword, question):
    keyword = str(keyword).casefold()
    question = str(question).casefold()
    if re.search(r"[\u4e00-\u9fff]", keyword):
        return keyword in question
    escaped = re.escape(keyword)
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", question))


def load_application_index(path=APPLICATION_INDEX_PATH):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_application_synthesis_question(question):
    has_application = any(
        keyword_in_question(keyword, question)
        for values in APPLICATION_KEYWORDS.values()
        for keyword in values
    )
    has_route_intent = any(
        keyword_in_question(keyword, question)
        for keyword in [
            "合成路径",
            "路径",
            "路线",
            "怎么合成",
            "如何制备",
            "查询",
            "找到",
            "route",
            "path",
            "synthesis",
            "prepare",
            "make",
        ]
    )
    return has_application and has_route_intent


def requested_applications(question):
    applications = []
    for application, keywords in APPLICATION_KEYWORDS.items():
        if any(keyword_in_question(keyword, question) for keyword in keywords):
            applications.append(application)
    return applications


def material_matches_question(material, entry, question):
    question_text = str(question)
    labels = [material]
    labels.extend(entry.get("aliases", []))
    labels.extend([alias.replace("TBHP", "THBP") for alias in entry.get("aliases", [])])
    for label in labels:
        normalized_label = normalize(label)
        if not normalized_label or len(normalized_label) < 3:
            continue
        if re.fullmatch(r"[a-z0-9]+", normalized_label):
            if keyword_in_question(normalized_label, question_text):
                return True
        elif normalized_label in normalize(question_text):
            return True
    return False


def has_kinetics_route(entry):
    return any(
        route.get("route_source") == "kinetics_experimental_route"
        for route in entry.get("synthesis_routes", [])
    )


def select_materials(index, question, applications):
    explicit = [
        (material, entry)
        for material, entry in index.items()
        if material_matches_question(material, entry, question)
    ]
    if explicit:
        return explicit

    selected = []
    for material, entry in index.items():
        material_apps = set(entry.get("applications", []))
        if not applications or material_apps.intersection(applications):
            selected.append((material, entry))
    selected.sort(
        key=lambda item: (
            -int(has_kinetics_route(item[1])),
            -len(item[1].get("synthesis_routes", [])),
            item[0].casefold(),
        )
    )
    return selected


def is_chinese(question):
    return bool(re.search(r"[\u4e00-\u9fff]", str(question)))


def format_route(route, chinese=False):
    reactants = ", ".join(route.get("reactants", [])) or "not parsed"
    label = route.get("route_label") or route.get("product", "")
    source = route.get("route_source", "")
    source_file = route.get("source_file", "")
    if chinese:
        return f"  - {label}：反应物={reactants}；来源={source}；文件={source_file}"
    return f"  - {label}: reactants={reactants}; source={source}; file={source_file}"


def answer_application_synthesis_question(question, index=None, max_materials=6, max_routes=5):
    index = index or load_application_index()
    if not index or not is_application_synthesis_question(question):
        return ""

    applications = requested_applications(question)
    materials = select_materials(index, question, applications)
    chinese = is_chinese(question)
    if chinese:
        lines = [
            "可以。当前通用 OR QA 先查物质的应用标签，再返回这些物质的合成路径候选。",
            "查询应用: " + (", ".join(applications) if applications else "all"),
            f"命中物质数: {len(materials)}",
            "候选合成路径:",
        ]
    else:
        lines = [
            "Yes. The generic OR QA first searches material application labels, then returns synthesis-route candidates for matched materials.",
            "Requested applications: " + (", ".join(applications) if applications else "all"),
            f"Matched materials: {len(materials)}",
            "Candidate synthesis routes:",
        ]

    for material, entry in materials[:max_materials]:
        applications_text = ", ".join(entry.get("applications", []))
        aliases_text = ", ".join(entry.get("aliases", [])[:6])
        if chinese:
            lines.append(f"- 物质={material}；应用={applications_text}；别名={aliases_text}")
        else:
            lines.append(f"- material={material}; applications={applications_text}; aliases={aliases_text}")
        for route in entry.get("synthesis_routes", [])[:max_routes]:
            lines.append(format_route(route, chinese=chinese))

    if chinese:
        lines.append(
            "后续人工筛选：从这些候选中按目标物、实验可行性、时间序列数据、PINN结果质量筛出最终4条路径。"
        )
    else:
        lines.append(
            "Manual screening step: select the final four routes from these candidates by target relevance, feasibility, time-course data availability, and PINN fit quality."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    examples = [
        "有氧化剂和引发剂应用的物质有哪些合成路径？",
        "THBP和TBPB作为氧化剂、引发剂有哪些合成路径？",
        "Find synthesis routes for materials used as oxidants or initiators.",
    ]
    for example in examples:
        print("\nQuestion:", example)
        print(answer_application_synthesis_question(example))
