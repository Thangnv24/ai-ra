"""Prompt templates for constrained LLM entity discovery."""

from __future__ import annotations

import json
from typing import Any, Iterable

from core.config import (
    ALLOWED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)


ENTITY_SYSTEM_PROMPT = """You extract an exhaustive list of medical mentions from Vietnamese and English clinical text.
Return valid JSON only. Copy every quote exactly from target_units.
Do not infer diagnoses, normalize text, calculate offsets, assign assertions, or map ontology codes."""


def build_entity_extraction_prompt(
    chunk: dict[str, Any],
    concept_type: str | None = None,
    examples: Iterable[dict[str, str]] = (),
) -> str:
    """Build one compact multi-type prompt.

    ``concept_type`` and ``examples`` remain accepted for compatibility with
    preparation tools, but runtime discovery intentionally uses all types and
    no cross-dataset examples.
    """

    requested_types = [concept_type] if concept_type in ALLOWED_TYPES else list(ALLOWED_TYPES)
    type_rules = {
        TYPE_SYMPTOM: "complaints, signs, examination findings, mental status, functional findings, and explicit normal or negated findings",
        TYPE_DIAGNOSIS: "only diagnoses explicitly stated by a clinician or listed in diagnosis, history, or problem-list context; never infer from evidence",
        TYPE_TEST_NAME: "assays, measurements, procedures, imaging, pathology, endoscopy, ECG, and concrete abbreviated test names",
        TYPE_TEST_RESULT: "numeric values and complete qualitative findings tied to a test; keep a visible unit only when it belongs to that value phrase",
        TYPE_DRUG: "medicines with attached strength or dose form; keep route, frequency, and PRN only in English medication-list rows",
    }
    payload = {
        "task": "Extract every explicit medical mention occurrence from target_units.",
        "allowed_types": requested_types,
        "rules": [
            "Be exhaustive: include repeated occurrences, short clinical findings, negated mentions, and historical mentions.",
            "Return only exact quotes from target_units. Never quote context_before or context_after.",
            "Use unit_id and zero-based occurrence_index to identify the occurrence inside that unit.",
            "Do not return section headings, demographics, dates, identifiers, or administrative text.",
            *(f"{entity_type}: {type_rules[entity_type]}." for entity_type in requested_types),
            "Keep meaningful severity, anatomy, location, laterality, temporal, and exertional modifiers in the same mention.",
            "Split independent diagnoses or findings, but keep one complete qualitative test-report finding together.",
            "When both a test name and its result are visible, return them as separate mentions.",
        ],
        "context": {
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "subsection": chunk.get("subsection", "document"),
            "context_before": chunk.get("context_before", ""),
            "context_after": chunk.get("context_after", ""),
        },
        "target_units": chunk.get("units") or [
            {"unit_id": chunk["chunk_id"], "text": chunk["text"]}
        ],
        "response_contract": {
            "root": {"mentions": "array"},
            "mention_fields": {
                "unit_id": "exact unit_id from target_units",
                "quote": "exact substring from that unit",
                "type": {"allowed_values": requested_types},
                "occurrence_index": "zero-based integer for that exact quote in the unit",
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False)
