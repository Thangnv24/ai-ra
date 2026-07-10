"""Prompt templates for constrained LLM decisions."""

from __future__ import annotations

import json
from typing import Any

from core.config import ALLOWED_ASSERTIONS, ALLOWED_TYPES, TYPE_DIAGNOSIS, TYPE_SYMPTOM


SYSTEM_PROMPT = """You are a medical concept decision helper.
Return valid JSON only.
Only choose among provided mention proposals and retrieved candidates.
Do not invent ICD-10 or RxNorm codes.
selected_candidates must be a subset of the retrieved candidate codes.
Never change mention offsets."""


def build_decision_prompt(text: str, mentions: list[dict[str, Any]]) -> str:
    payload = {
        "task": "Decide whether to keep mentions, fix type/assertions if needed, and rerank candidates.",
        "rules": [
            "Never change mention offsets.",
            "selected_candidates must be a subset of retrieved candidate codes.",
            "Only diagnosis and drug mentions may have selected_candidates.",
            "Symptoms, lab names, and lab results must use selected_candidates: [].",
            "Use assertions only for symptoms, diagnoses, and drugs.",
            "Prefer high precision over adding uncertain mentions.",
        ],
        "allowed_types": list(ALLOWED_TYPES),
        "allowed_assertions": list(ALLOWED_ASSERTIONS),
        "document": text,
        "mentions": mentions,
        "response_schema": {
            "decisions": [
                {
                    "mention_id": "m1",
                    "keep": True,
                    "final_type": TYPE_DIAGNOSIS,
                    "assertions": ["isHistorical"],
                    "selected_candidates": ["I10"],
                    "confidence": 0.92,
                    "reason": "short log-only reason",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)


ENTITY_SYSTEM_PROMPT = """You extract medical mentions from Vietnamese clinical text.
Return valid JSON only.
Every quote must be an exact substring copied from the provided chunk.
Do not invent normalized text, offsets, ICD-10 codes, or RxNorm codes."""


def build_entity_extraction_prompt(chunk: dict[str, Any]) -> str:
    payload = {
        "task": "Extract medical mention quotes from this chunk.",
        "rules": [
            "Return exact quotes only; each quote must appear verbatim in chunk_text.",
            "Use only the allowed_types.",
            "Do not return patient demographics, age, address, phone, dates, or section headings.",
            "Do not return ICD-10/RxNorm candidates.",
            "Prefer complete clinical phrases over single generic words.",
            "For labs, return test names and numeric results separately when visible.",
        ],
        "allowed_types": list(ALLOWED_TYPES),
        "allowed_assertions": list(ALLOWED_ASSERTIONS),
        "chunk": {
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "chunk_text": chunk["text"],
        },
        "response_schema": {
            "mentions": [
                {
                    "quote": "ho đờm xanh",
                    "type": TYPE_SYMPTOM,
                    "confidence": 0.92,
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)
