"""Prompt templates for constrained LLM discovery."""

from __future__ import annotations

import json
from typing import Any, Iterable

from core.config import (
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)


ENTITY_SYSTEM_PROMPT = """You are an annotation candidate discoverer for a medical span-labeling dataset.
Follow the organizer's annotation policy, not general clinical reasoning.
Return valid JSON only. Copy every quote exactly from one target unit.
Never infer a diagnosis, rewrite text, calculate offsets, map ontology codes, or assign assertions.
When text is merely plausible but not an explicit annotation unit, omit it."""


_TYPE_POLICY: dict[str, list[str]] = {
    TYPE_SYMPTOM: [
        "Find explicit patient complaints, signs, examination findings, vital signs, mental status, and functional findings.",
        "Keep attached location, laterality, severity, temporal, and exertional modifiers when they form one finding.",
        "Do not reduce a specific finding to a generic head word.",
        "A negation cue may be part of the quote when it is grammatically attached; assertions are assigned later.",
        "Reject isolated generic words such as abnormal, increase, decrease, condition, or finding unless the unit states a complete clinical finding.",
    ],
    TYPE_DIAGNOSIS: [
        "Find only diagnoses explicitly stated by a clinician or listed in diagnosis, history, or problem-list context.",
        "Do not infer a disease from symptoms, tests, drugs, or medical knowledge.",
        "Prefer the named disease unit. Do not append explanatory prose or a neighboring independent diagnosis.",
        "Keep stage, severity, anatomy, or cause only when it is part of the named diagnosis in this template.",
        "A surface form that is only an observed finding is not a diagnosis outside explicit diagnosis context.",
    ],
    TYPE_TEST_NAME: [
        "Find explicit assay, procedure, imaging, pathology, endoscopy, and physical-measurement names.",
        "Prefer the complete visible test label, including aliases or method text when they belong to that label.",
        "Do not emit a nested abbreviation such as PT or TQ when it occurs inside one complete test label.",
        "Do not emit the generic word test or examination without a concrete procedure role.",
        "Do not include the result value in a test-name quote.",
    ],
    TYPE_TEST_RESULT: [
        "Find explicit numeric values with visible units and complete qualitative findings tied to a test or measurement.",
        "Keep a visible unit with its value.",
        "Keep one complete qualitative report finding instead of detached adjectives.",
        "Reject unanchored numbers, dates, identifiers, doses, and demographic values.",
        "Do not include the test name unless the local template expresses the full label-plus-value as one vital-sign finding.",
    ],
    TYPE_DRUG: [
        "Find medicines explicitly used, prescribed, administered, or recorded in medication history.",
        "A poison, recreational substance, pesticide, or exposure chemical is not a medicine merely because it has a chemical name.",
        "Keep brand or ingredient and attached strength or dose form.",
        "In prescription templates, avoid count and administration instructions that are outside the medicine unit.",
        "When the source is an English medication-list template, route, frequency, and PRN may belong to the medicine unit; copy only the convention supported by the local row.",
    ],
}


_CONTRASTS: dict[str, list[dict[str, str]]] = {
    TYPE_SYMPTOM: [
        {"input": "không đau đầu", "prefer": "không đau đầu", "avoid": "đau đầu"},
        {"input": "đau ngực trái", "prefer": "đau ngực trái", "avoid": "đau ngực"},
    ],
    TYPE_DIAGNOSIS: [
        {
            "input": "Viêm tụy cấp Balthazar D, do sỏi đường mật",
            "prefer": "Viêm tụy cấp Balthazar D",
            "avoid": "the whole explanatory clause",
        },
        {"input": "men gan tăng, nghĩ viêm gan", "prefer": "viêm gan", "avoid": "inference from men gan tăng"},
    ],
    TYPE_TEST_NAME: [
        {
            "input": "Thời gian prothrombin (PT; TQ) bằng máy tự động",
            "prefer": "the complete visible label",
            "avoid": "PT or TQ alone",
        },
        {"input": "xét nghiệm cho thấy bất thường", "prefer": "omit", "avoid": "xét nghiệm"},
    ],
    TYPE_TEST_RESULT: [
        {"input": "Natri: 132 mmol/L", "prefer": "132 mmol/L", "avoid": "132"},
        {"input": "ngày 12/07/2026", "prefer": "omit", "avoid": "12/07/2026"},
    ],
    TYPE_DRUG: [
        {"input": "Exforge 5/80mg x 1 viên sáng", "prefer": "Exforge 5/80mg", "avoid": "x 1 viên sáng"},
        {"input": "ngộ độc Glufosinate", "prefer": "omit", "avoid": "Glufosinate"},
    ],
}


def build_entity_extraction_prompt(
    chunk: dict[str, Any],
    concept_type: str,
    examples: Iterable[dict[str, str]] = (),
) -> str:
    if concept_type not in _TYPE_POLICY:
        raise ValueError(f"Unsupported entity type: {concept_type!r}")
    retrieved_examples = [
        {
            "left_context": str(item.get("left") or ""),
            "correct_quote": str(item.get("quote") or ""),
            "right_context": str(item.get("right") or ""),
            "section": str(item.get("section") or ""),
            "subsection": str(item.get("subsection") or ""),
        }
        for item in examples
        if str(item.get("quote") or "")
    ]
    payload = {
        "task": f"Discover explicit candidate spans of exactly one type: {concept_type}.",
        "requested_type": concept_type,
        "common_rules": [
            "Return every explicit occurrence of the requested type, including repeated occurrences.",
            "Do not return another entity type in this call.",
            "Each quote must be copied verbatim from exactly one target unit.",
            "context_before and context_after are context only and may never be quoted.",
            "Use unit_id to identify the target unit and occurrence_index to distinguish repeated exact quotes inside that unit.",
            "Do not return offsets, confidence, assertions, ICD-10, or RxNorm.",
            "Do not diagnose, normalize, summarize, or complete missing text.",
            "Return an empty mentions list when there is no explicit annotation unit of the requested type.",
        ],
        "type_policy": _TYPE_POLICY[concept_type],
        "contrastive_examples": _CONTRASTS[concept_type],
        "retrieved_positive_examples": retrieved_examples[:4],
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
        "response_schema": {
            "mentions": [
                {
                    "unit_id": "exact unit_id from target_units",
                    "quote": "exact substring copied from that unit",
                    "occurrence_index": 0,
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)
