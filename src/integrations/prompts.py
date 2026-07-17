"""Prompt templates for constrained LLM decisions."""

from __future__ import annotations

import json
from typing import Any

from core.config import (
    ALLOWED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)


ENTITY_SYSTEM_PROMPT = """You extract an exhaustive, dense list of medical mentions from Vietnamese clinical text.
Return valid JSON only.
Every quote must be an exact substring copied from target_text.
Do not invent normalized text, offsets, ICD-10 codes, or RxNorm codes."""


def build_entity_extraction_prompt(chunk: dict[str, Any]) -> str:
    payload = {
        "task": "Extract medical mention quotes from this chunk.",
        "rules": [
            "Be exhaustive. Do not summarize the case or return only the most important diagnoses; list every visible medical mention occurrence.",
            "Return every medical mention occurrence separately, including repeated text at different positions.",
            "Extract only from target_text. context_before and context_after are classification context and must never be quoted.",
            "Return exact quotes only; each quote must appear verbatim in target_text.",
            "occurrence_index is zero-based among exact matches of the same quote inside target_text.",
            "Use only the allowed_types.",
            "Classify every occurrence from its role in the case, nearby context, and target text, not from the term alone.",
            f"Use {TYPE_SYMPTOM} for every patient clinical finding or observation, including complaints, signs, vital signs, mental or functional status, and normal or negated examination findings.",
            f"Do include short standalone {TYPE_SYMPTOM} mentions when the text uses them as clinical findings, such as đau, ngã, phù, yếu, sốt, ho, ngất, run rẩy, tê bì, tự sát, chán ăn.",
            f"Use {TYPE_TEST_NAME} for the procedure or assay name, and {TYPE_TEST_RESULT} for its numeric or qualitative finding.",
            f"Do include generic or abbreviated {TYPE_TEST_NAME} mentions when they are clinical measurements/procedures, such as xét nghiệm, CTM, điện tim, siêu âm, chụp cắt lớp vi tính, huyết áp, nhịp thở.",
            f"Keep a numeric {TYPE_TEST_RESULT} together with its visible unit.",
            f"A {TYPE_TEST_RESULT} may be a plain number, sign, short qualitative token, or full sentence when it is explicitly the value/finding of a test, vital sign, imaging study, ECG, pathology, or physical measurement.",
            f"Keep one complete qualitative imaging, ECG, endoscopy, pathology, or biopsy report block as {TYPE_TEST_RESULT}, including line breaks when needed.",
            f"Use {TYPE_DIAGNOSIS} when a clinician states a diagnosis or the term appears in a diagnosis/history/problem list.",
            f"Do include common {TYPE_DIAGNOSIS} and problem-list terms, such as trầm cảm, rối loạn lipid máu, rung nhĩ, tai biến mạch máu não, bệnh mạch vành, suy giáp, bệnh thận mạn.",
            "The same quote may have different types at different positions; return each occurrence with its contextual type.",
            "Do not return patient demographics, age, address, phone, dates, or section headings.",
            "Do not return ICD-10/RxNorm candidates.",
            "Use the exact complete phrase used as one clinical finding; do not shorten away severity, location, result units, or report details.",
            "If the text lists many independent medical terms in one sentence, split them into separate mentions unless the words form one single named condition, drug, test, or report finding.",
            "Split simple lists of independent diagnoses or findings, but do not split a single qualitative test report block.",
            "For drugs, keep visible brand or ingredient, strength, dose form, route, frequency, and PRN modifiers with the drug name.",
            "For labs, return test names and results separately when both are visible.",
            "Include negated and historical mentions; assertion labels are assigned later.",
        ],
        "allowed_types": list(ALLOWED_TYPES),
        "target": {
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "context_before": chunk.get("context_before", ""),
            "target_text": chunk["text"],
            "context_after": chunk.get("context_after", ""),
        },
        "response_schema": {
            "mentions": [
                ["ho đờm xanh", TYPE_SYMPTOM, 0]
            ]
        },
        "mention_format": ["exact_quote", "type", "occurrence_index"],
    }
    return json.dumps(payload, ensure_ascii=False)
