"""Prompt templates for constrained local LLM reranking."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a medical concept decision helper.
Only choose among provided mention proposals and retrieved candidates.
Return valid JSON only. Do not invent ICD-10 or RxNorm codes.
selected_candidates must be a subset of the retrieved candidate codes.
Never change mention offsets."""


def build_decision_prompt(text: str, mentions: list[dict[str, Any]]) -> str:
    payload = {
        "task": "Decide whether to keep mentions, fix type/assertions if needed, and rerank candidates.",
        "allowed_types": ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC"],
        "allowed_assertions": ["isNegated", "isFamily", "isHistorical"],
        "document": text,
        "mentions": mentions,
        "response_schema": {
            "decisions": [
                {
                    "mention_id": "m1",
                    "keep": True,
                    "final_type": "CHẨN_ĐOÁN",
                    "assertions": ["isHistorical"],
                    "selected_candidates": ["I10"],
                    "confidence": 0.92,
                    "reason": "short log-only reason",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)
