import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
APPLICATION_KEYWORDS = {
    "oxidant": (
        "oxidant", "oxidants", "oxidizer", "oxidizers", "oxidiser", "oxidisers",
        "oxidizing agent", "oxidizing agents", "oxidising agent", "oxidising agents",
        "\u6c27\u5316\u5242",
    ),
    "initiator": (
        "initiator", "initiators", "\u5f15\u53d1\u5242",
    ),
    "solvent": ("solvent", "solvents", "\u6eb6\u5242"),
    "acid_catalyst": ("acid catalyst", "acid catalysts", "\u9178\u50ac\u5316\u5242"),
    "base": ("base", "bases", "\u78b1"),
    "acylating_agent": ("acylating agent", "acylating agents", "\u9170\u5316\u5242"),
    "substrate": ("substrate", "substrates", "\u5e95\u7269"),
}
CHEMICAL_ALIASES = {
    "TBHP": ("t-butyl hydroperoxide", "\u53d4\u4e01\u57fa\u8fc7\u6c27\u5316\u6c22", "\u8fc7\u6c27\u5316\u6c22\u53d4\u4e01\u57fa"),
    "TBPB": ("t-butyl perbenzoate", "t-butyl peroxybenzoate", "\u8fc7\u6c27\u5316\u82ef\u7532\u9178\u53d4\u4e01\u916f", "\u8fc7\u82ef\u7532\u9178\u53d4\u4e01\u916f"),
}
SYNTHESIS_TERMS = (
    "synthesis", "syntheses", "synthesize", "synthesise", "synthetic",
    "synthesized", "synthesised", "synthesizing", "synthesising",
    "prepare", "prepared", "preparing", "preparation", "make", "made",
    "manufacture", "manufacturing", "produce", "produced", "production",
    "\u5408\u6210", "\u5236\u5907", "\u751f\u4ea7",
)
ROUTE_TERMS = ("route", "routes", "path", "paths", "pathway", "pathways", "\u8def\u7ebf", "\u8def\u5f84")
KINETIC_TERMS = (
    "kinetics", "kinetic", "rate", "rates", "multiplier", "multipliers",
    "activation energy", "activation energies", "reaction order", "reaction orders",
    "yield", "yields", "selectivity", "selectivities", "conversion", "conversions",
    "concentration", "concentrations", "temperature", "temperatures",
    "residence time", "screen", "screening", "optimum",
    "\u52a8\u529b\u5b66", "\u901f\u7387", "\u6d3b\u5316\u80fd", "\u6536\u7387", "\u9009\u62e9\u6027",
    "\u8f6c\u5316\u7387", "\u6d53\u5ea6", "\u6e29\u5ea6", "\u505c\u7559\u65f6\u95f4",
)


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value).strip()


def keyword_matches(keyword, question):
    keyword = normalize(keyword)
    if re.search(r"[\u4e00-\u9fff]", keyword):
        return keyword in question
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])", question))


def contains_any(terms, question):
    return any(keyword_matches(term, question) for term in terms)


def material_mentions(index, question):
    matches = []
    for material, entry in index.items():
        labels = [material, entry.get("material", material), *entry.get("aliases", [])]
        for acronym, aliases in CHEMICAL_ALIASES.items():
            if acronym in labels:
                labels.extend(aliases)
        for label in set(labels):
            label = normalize(label)
            if len(label) < 3:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(label) + r"(?![a-z0-9])"
            for match in re.finditer(pattern, question):
                matches.append((material, match.start(), match.end()))
    found = set()
    for material, start, end in matches:
        if not any(left <= start and end <= right and right - left > end - start for _, left, right in matches):
            found.add(material)
    return found


def load_sources(path):
    sources = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row["source_id"]
            if source_id in sources:
                raise ValueError("Duplicate application source ID: " + source_id)
            sources[source_id] = row
    return sources


def rank_material(item):
    material, entry = item
    routes = entry.get("synthesis_routes", [])
    experimental = any(route.get("route_source") == "kinetics_experimental_route" for route in routes)
    return -int(experimental), -len(routes), material.casefold()


def unmatched_acronyms(question, index):
    known = {
        token
        for material, entry in index.items()
        for label in [material, *entry.get("aliases", [])]
        for token in normalize(label).split()
    }
    common = set("a an and any are as at be by can find for from how i in is it list make of on or show synthesis the to used uses use what which with routes route materials material compounds compound please synthesis initiators initiator oxidants oxidant oxidizers oxidizer oxidising oxidizing agents agent".split())
    return sorted({
        token for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,11}(?![A-Za-z0-9])", str(question))
        if normalize(token) not in known and token.casefold() not in common
    })


def unmatched_target(question, explicit):
    if explicit:
        return []
    match = re.search(r"\b(?:synthesize|synthesise|prepare|make|manufacture|(?:synthesis(?: routes?)?|routes?) (?:of|for|to)) (.+)", question)
    if match is None:
        match = re.search(r"\b(?:how (?:can|could)|can|could) (.+?) be (?:synthesized|synthesised|prepared|made|produced)\b", question)
    if match is None:
        match = re.search(r"^([a-z0-9 ]+?) (?:as|used as) (?:an? )?(?:initiator|oxidant|oxidizer|oxidiser)\b", question)
    if match is None:
        return []
    target = re.split(r"\b(?:as|used as|for use as|which|that|with|under|in)\b", match.group(1), maxsplit=1)[0].strip()
    general = set("a an the any these those different various suitable common potential organic inorganic peroxide peroxides material materials compound compounds chemical chemicals substance substances agent agents candidate candidates radical polymerization polymerisation and or".split())
    for terms in APPLICATION_KEYWORDS.values():
        for term in terms:
            general.update(normalize(term).split())
    return [target] if target and set(target.split()) - general else []


def retrieve_application_routes(question, index_path=None, sources_path=None, max_materials=6, max_routes=5):
    if max_materials < 1 or max_routes < 1:
        raise ValueError("Material and route limits must be positive")
    question_text = normalize(question)
    applications = [key for key, terms in APPLICATION_KEYWORDS.items() if contains_any(terms, question_text)]
    synthesis_intent = contains_any(SYNTHESIS_TERMS, question_text)
    route_intent = contains_any(ROUTE_TERMS, question_text)
    numbered_route = bool(re.search(r"\b(?:path|route)s?\s*[1-4]\b", question_text))
    if contains_any(KINETIC_TERMS, question_text) and (numbered_route or not synthesis_intent):
        return None
    index_path = Path(index_path) if index_path is not None else DATA / "material_application_index.json"
    sources_path = Path(sources_path) if sources_path is not None else DATA / "application_route_sources.jsonl"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    explicit = material_mentions(index, question_text)
    unknown_acronyms = unmatched_acronyms(question, index)
    unknown_targets = unmatched_target(question_text, explicit)
    bare_material = any(
        question_text == normalize(label)
        for material in explicit
        for label in [material, *index[material].get("aliases", []), *sum(CHEMICAL_ALIASES.values(), ())]
    )
    if not applications and not synthesis_intent and not route_intent and not bare_material:
        return None
    if not applications and not explicit and not synthesis_intent:
        return None
    selected = [
        (material, entry)
        for material, entry in index.items()
        if material in explicit or (not explicit and not unknown_acronyms and not unknown_targets and applications and set(entry.get("applications", [])).intersection(applications))
    ]
    selected.sort(key=rank_material)
    sources = load_sources(sources_path)
    records = []
    used_sources = {}
    for material, entry in selected[:max_materials]:
        available_routes = [route for route in entry.get("synthesis_routes", []) if route.get("retrieval_status") != "source_role_conflict"]
        routes = available_routes[:max_routes]
        for route in routes:
            source_id = route["source_id"]
            if source_id not in sources:
                raise ValueError("Missing application route source: " + source_id)
            used_sources[source_id] = sources[source_id]
        fields = dict(entry)
        fields["synthesis_routes"] = routes
        fields["matched_applications"] = sorted(set(entry.get("applications", [])).intersection(applications))
        fields["available_route_count"] = len(available_routes)
        fields["indexed_route_count"] = len(entry.get("synthesis_routes", []))
        records.append({"id": "material:" + material, "kind": "Material", "fields": fields})
    return {
        "records": records,
        "sources": list(used_sources.values()),
        "calculations": [],
        "retrieval_method": "application_route_index",
        "matching_mode": "material_alias" if explicit else "application_label" if applications else "no_material_match",
        "application_match": "any",
        "matched_applications": applications,
        "unmatched_acronyms": unknown_acronyms,
        "unmatched_targets": unknown_targets,
        "matched_material_count": len(selected),
        "returned_material_count": len(records),
        "candidate_order": "experimental_route_presence, route_count, material_name",
        "application_label_scope": "Name rules and curated aliases provide candidate application labels; source records document route evidence.",
        "source_snapshot": "sha256:" + hashlib.sha256(index_bytes).hexdigest(),
        "source_records_sha256": hashlib.sha256(sources_path.read_bytes()).hexdigest(),
        "reference_answers_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--max-materials", type=int, default=6)
    parser.add_argument("--max-routes", type=int, default=5)
    args = parser.parse_args()
    result = retrieve_application_routes(args.question, args.index, args.sources, args.max_materials, args.max_routes)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
