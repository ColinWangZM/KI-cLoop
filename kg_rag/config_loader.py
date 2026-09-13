"""Load KGRAG settings without embedding machine-specific paths or secrets."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CONFIG_PATH = Path(
    os.environ.get("KGRAG_CONFIG", str(PACKAGE_DIR / "config.yaml"))
).expanduser()
SYSTEM_PROMPTS_PATH = Path(
    os.environ.get("KGRAG_SYSTEM_PROMPTS", str(PACKAGE_DIR / "system_prompts.yaml"))
).expanduser()

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config_data = yaml.safe_load(handle) or {}

with SYSTEM_PROMPTS_PATH.open("r", encoding="utf-8") as handle:
    system_prompts = yaml.safe_load(handle) or {}


PATH_KEYS = {
    "VECTOR_DB_DISEASE_ENTITY_PATH",
    "VECTOR_DB_ORGANIC_REACTION_PATH",
    "VECTOR_DB_PATH",
    "VECTOR_DB_PATH_OR",
    "NODE_CONTEXT_PATH",
    "NODE_CONTEXT_PATH_OR",
    "GPT_CONFIG_FILE",
    "LLM_CACHE_DIR",
    "SAVE_RESULTS_PATH",
    "MCQ_PATH",
    "MCQ_PATH_or",
    "OR_LORA_PATH",
}


def _portable_path(value: object) -> object:
    if not isinstance(value, str):
        return value
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str(PROJECT_ROOT / expanded)


for key in PATH_KEYS.intersection(config_data):
    config_data[key] = _portable_path(config_data[key])

__all__ = ["CONFIG_PATH", "PROJECT_ROOT", "config_data", "system_prompts"]
