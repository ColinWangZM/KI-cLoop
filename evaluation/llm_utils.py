"""Shared utilities for LLM evaluation.

Import helpers from this module; it is not meant to be executed directly.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

OPENAI_BASE_URL = "https://api.openai.com/v1"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENAI_ENV = "OPENAI_API_KEY"
DASHSCOPE_ENV = "DASHSCOPE_API_KEY"
DEEPSEEK_ENV = "DEEPSEEK_API_KEY"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert assistant for materials science and chemistry questions."
    " Provide clear, concise answers."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial auto-grader for chemistry and materials questions."
    " Always return a strict JSON object with keys 'score' (0-10 scale) and 'justification' (concise text)."
    " Follow the provided scoring instructions and do not include any extra commentary outside the JSON object."
)

PROMPT_LIBRARY: Dict[str, Dict[str, str]] = {
    "Material Ontology Understanding": {
        "system": (
            "You are a senior materials scientist. Provide accurate, high-level explanations"
            " linking structure, properties, and applications."
        ),
        "user_suffix": "Answer in 2-3 sentences, emphasizing the governing materials-science principles.",
    },
    "Attribute Query": {
        "system": (
            "You are a precise materials database assistant. Ensure attribute lookups stay consistent"
            " with the given data."
        ),
        "user_suffix": (
            "Respond with `Answer: <letter>. <short justification referencing the provided attributes>.`"
        ),
    },
    "Multi-Attribute Query": {
        "system": (
            "You analyze multiple criteria for materials datasets and choose the best matching option."
        ),
        "user_suffix": (
            "Respond with `Answer: <letter>. <brief rationale covering each required attribute>.`"
        ),
    },
    "Chain Reasoning": {
        "system": (
            "You are a senior synthetic chemist. Reconstruct multi-step reaction pathways using precise language,"
            " naming intermediates, reagents, and operations at each stage. Highlight causal links between steps"
            " and end with a concise overall summary."
        ),
        "user_suffix": (
            "Format your answer as:\n"
            "Step 1 Reactants: <reactants>; Products: <products>; Operation Steps: <key operations and conditions>.\n"
            "Step 2  … (continue for each step)\n"
            "Complete Pathway: <one or two sentences capturing the net transformation>.\n\n"
            "Emphasize reagent roles, isolation or purification operations, and how each step enables the next."
            " Avoid inventing additional steps or conditions beyond the provided information."
        ),
    },
    "Conditional Reasoning": {
        "system": (
            "You interpret experimental conditions and predict outcomes based on quantitative trends."
        ),
        "user_suffix": "State the predicted outcome in one concise sentence.",
    },
    "Multi-path Comparison": {
        "system": (
            "You compare alternative synthetic pathways, highlighting trade-offs before deciding on the best option."
        ),
        "user_suffix": (
            "Compare the candidate pathways first, then conclude with `Final verdict: <choice>` summarizing the best option."
        ),
    },
}

DEFAULT_JUDGE_INSTRUCTIONS = (
    "Compare the candidate answer against the reference answer. Score from 0 (completely incorrect) to 10"
    " (fully correct and complete). Award intermediate scores for partially correct responses."
    " Keep the justification brief and reference concrete mismatches or alignments."
)

JUDGE_PROMPT_LIBRARY: Dict[str, Dict[str, object]] = {
    "Material Ontology Understanding": {
        "system": (
            "You are an impartial evaluator with expertise in materials science terminology and ontology."
            " Judge the conceptual quality of answers using your own knowledge when no reference answer is provided."
        ),
        "instructions": (
            "No reference answer is supplied. Score from 0 to 10 based on how accurately and comprehensively the"
            " candidate explains the materials ontology concepts in the question."
            " Reward precise terminology, clear structure-property reasoning, and coverage of key facets."
            " Penalize factual mistakes, missing rationales, or vague statements."
        ),
        "require_reference": False,
    },
    "Chain Reasoning": {
        "instructions": (
            "Compare each step of the candidate's chain-of-thought to the reference answer."
            " Award 10 only if all reaction stages, intermediates, and causal links match."
            " Deduct points for missing steps, incorrect reagents, wrong sequencing, or unsupported conclusions."
        ),
    },
    "Multi-path Comparison": {
        "instructions": (
            "Judge how well the candidate contrasts the pathways versus the reference answer and whether the final"
            " verdict aligns. Award 10 for complete, well-reasoned comparisons that match the reference decision."
            " Deduct points for missing trade-offs, incorrect rationale, or wrong conclusion."
        ),
    },
    "Conditional Reasoning": {
        "instructions": (
            "Compare the candidate's predicted outcome with the reference answer. Score from 0 to 10 based on"
            " correctness, clarity, and whether the response captures the quantitative trend implied by the"
            " question. Reward concise, accurate predictions; penalize incorrect outcomes or missing trend"
            " interpretation."
        ),
    },
}

REFERENCE_OPTIONAL_TYPES = {
    question_type
    for question_type, config in JUDGE_PROMPT_LIBRARY.items()
    if not config.get("require_reference", True)
}


@dataclass
class Task:
    dataset: str
    question_id: str
    question_type: str
    question: str
    selection: Optional[str]
    reference_answer: Optional[str]
    correct_choice: Optional[str]


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float = 0.2,
        request_timeout: int = 60,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable '{api_key_env}' for provider '{provider}'."
            )
        self.provider = provider
        self.model = model
        self.temperature = temperature
        base = base_url.rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=base).with_options(timeout=request_timeout)
        self._force_default_temperature = False

    def complete(self, messages: List[Dict[str, str]], max_output_tokens: int = 512) -> str:
        def _should_retry_with_completion_tokens(message: str) -> bool:
            return "max_tokens" in message and "max_completion_tokens" in message

        def _should_retry_without_temperature(message: str) -> bool:
            return "temperature" in message and "only the default" in message.lower()

        use_completion_tokens = False

        while True:
            params: Dict[str, object] = {
                "model": self.model,
                "messages": messages,
            }
            if not self._force_default_temperature and self.temperature is not None:
                params["temperature"] = self.temperature
            token_param = "max_completion_tokens" if use_completion_tokens else "max_tokens"
            params[token_param] = max_output_tokens

            try:
                response = self.client.chat.completions.create(**params)
                break
            except Exception as exc:  # noqa: BLE001
                message = str(getattr(exc, "message", "")) or str(exc)
                if self.provider == "openai":
                    if not use_completion_tokens and _should_retry_with_completion_tokens(message):
                        use_completion_tokens = True
                        continue
                    if (not self._force_default_temperature) and _should_retry_without_temperature(message):
                        self._force_default_temperature = True
                        continue
                raise RuntimeError(f"{self.provider} {self.model} request failed: {exc}") from exc

        try:
            return (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response payload from {self.provider}: {response}") from exc


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return list(reader)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Failed to decode CSV '{csv_path}' with encodings {encodings}."
    ) from last_error


def load_tasks(dataset_dir: Path, require_reference: bool = True) -> List[Task]:
    tasks: List[Task] = []
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        rows = _read_csv_rows(csv_path)
        for row in rows:
            reference = (row.get("Answer") or "").strip() or None
            if require_reference and not reference:
                continue
            task = Task(
                dataset=csv_path.name,
                question_id=(row.get("QuestionID") or "").strip(),
                question_type=(row.get("QuestionType") or "").strip(),
                question=(row.get("Question") or "").strip(),
                selection=(row.get("Selection") or "").strip() or None,
                reference_answer=reference,
                correct_choice=(row.get("True_ANS") or "").strip() or None,
            )
            tasks.append(task)
    return tasks


def build_candidate_messages(
    task: Task,
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    config = PROMPT_LIBRARY.get(task.question_type or "", {})
    system_prompt = config.get("system", default_system_prompt)

    parts = [f"Question ID: {task.question_id}", task.question]
    if task.selection:
        parts.append("Options: " + task.selection)

    user_suffix = config.get("user_suffix")
    if user_suffix:
        parts.append(user_suffix)
    elif task.selection:
        parts.append("Provide the correct option letter followed by a brief explanation.")
    else:
        parts.append("Provide a concise, well-structured answer.")

    user_prompt = "\n\n".join(parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_judge_messages(task: Task, candidate_answer: str) -> List[Dict[str, str]]:
    config = JUDGE_PROMPT_LIBRARY.get(task.question_type or "", {})
    system_prompt = config.get("system", JUDGE_SYSTEM_PROMPT)
    instructions = config.get("instructions", DEFAULT_JUDGE_INSTRUCTIONS)

    payload = {
        "question_id": task.question_id,
        "question": task.question,
        "options": task.selection,
        "reference_answer": task.reference_answer,
        "correct_choice": task.correct_choice,
        "candidate_answer": candidate_answer,
        "question_type": task.question_type,
        "max_score": 10,
        "scoring_guidelines": instructions,
    }
    judge_request = json.dumps(payload, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{instructions}\n\nEvaluate the following response using a 0-10 scale and return only JSON with"
                f" keys 'score' and 'justification'.\n```json\n{judge_request}\n```"
            ),
        },
    ]


def parse_judge_output(text: str) -> Dict[str, str]:
    print(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Judge response is not JSON: {text}")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def create_client(descriptor: str, temperature: float) -> OpenAICompatibleClient:
    try:
        provider, model = descriptor.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"Model descriptor '{descriptor}' must use the format 'provider:model'."
        ) from exc
    provider = provider.lower()
    if provider == "openai":
        base_url = OPENAI_BASE_URL
        api_key_env = OPENAI_ENV
    elif provider == "qwen":
        base_url = DASHSCOPE_BASE_URL
        api_key_env = DASHSCOPE_ENV
    elif provider == "deepseek":
        base_url = DEEPSEEK_BASE_URL
        api_key_env = DEEPSEEK_ENV
    else:
        base_url = DASHSCOPE_BASE_URL
        api_key_env = DASHSCOPE_ENV
    return OpenAICompatibleClient(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=temperature,
    )
