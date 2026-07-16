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


ENTITY_SYSTEM_PROMPT = """You extract medical mentions from Vietnamese clinical text.
Return valid JSON only.
Every quote must be an exact substring copied from the selected unit_text.
Do not invent normalized text, offsets, ICD-10 codes, or RxNorm codes."""


def build_entity_extraction_prompt(chunk: dict[str, Any]) -> str:
    payload = {
        "task": "Extract medical mention quotes from this chunk.",
        "rules": [
            "Return every medical mention occurrence separately, including repeated text at different positions.",
            "Return exact quotes only; each quote must appear verbatim in the unit_text identified by unit_id.",
            "occurrence_index is zero-based among exact matches of the same quote inside that unit_text.",
            "Use only the allowed_types.",
            "Classify every occurrence from its role in the nearby units and section, not from the medical term alone.",
            f"Use {TYPE_TEST_NAME} for the procedure or assay name, and {TYPE_TEST_RESULT} for its numeric or qualitative finding.",
            f"Imaging, ECG, endoscopy, pathology, and biopsy findings are {TYPE_TEST_RESULT}, even when the finding is also a disease name.",
            f"Use {TYPE_DIAGNOSIS} when a clinician states a diagnosis or the term appears in a diagnosis/history/problem list.",
            "The same quote may have different types at different positions; return each occurrence with its contextual type.",
            "Do not return patient demographics, age, address, phone, dates, or section headings.",
            "Do not return ICD-10/RxNorm candidates.",
            "Return the shortest atomic phrase that still names the complete medical concept.",
            "Split coordinated concepts into separate mentions, for example two diagnoses joined by 'và'.",
            "For drugs, keep visible strength, dose form, route, frequency, and PRN modifiers with the drug name.",
            "For labs, return test names and numeric results separately when visible.",
            "Do not include trailing timing/severity clauses in symptom quotes when the core symptom appears as a clean substring.",
        ],
        "allowed_types": list(ALLOWED_TYPES),
        "chunk": {
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "units": chunk["units"],
        },
        "response_schema": {
            "mentions": [
                {
                    "unit_id": "c1u2",
                    "quote": "ho đờm xanh",
                    "occurrence_index": 0,
                    "type": TYPE_SYMPTOM,
                    "confidence": 0.92,
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)
