"""Prompt templates for constrained LLM decisions."""

from __future__ import annotations

import json
from typing import Any

from core.config import (
    ALLOWED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)


ENTITY_SYSTEM_PROMPT = """You are a high-precision span tagger for Vietnamese and English clinical text.
Return exactly one JSON object with a "mentions" array.
Every quote must be copied verbatim from target_text. Never invent text, offsets, or ontology codes."""


def _role_policy(structure_role: str) -> str:
    policies = {
        "question": (
            "Extract explicit facts about the described patient. Exclude diseases or tests that appear only "
            "as answer choices, differential suggestions, or things the question asks the reader to consider."
        ),
        "answer": (
            "Extract conclusions, findings, tests, and treatments explicitly applied to the described patient. "
            "Exclude general explanations, definitions, and unrelated examples."
        ),
        "medical_article": (
            "Extract only facts tied to a concrete patient or case in this block. Exclude general disease "
            "definitions, causes, risks, prevalence, possible complications, and illustrative examples."
        ),
        "family_history": (
            "Extract conditions explicitly attributed to a family member. Do not transfer them to the patient."
        ),
        "personal_history": "Extract explicit past conditions, symptoms, tests, and drugs.",
        "medical_history": "Extract explicit past conditions, symptoms, tests, and drugs.",
        "epidemiology_history": "Extract only explicit exposures, conditions, tests, and drugs.",
        "treatment": (
            "Prioritize complete medication orders and treatment-related tests or findings. Also tag any other "
            "concrete patient mention visible in this mixed block."
        ),
        "lab_or_imaging": (
            "Scan every line for paired test names and results, preserving visible units. Short abbreviations "
            "and positive/negative markers are eligible."
        ),
        "diagnosis": (
            "Prioritize explicit diagnoses and split independent diagnoses, but also tag drugs, tests, results, "
            "and findings if they are visibly present because a diagnosis block may contain mixed content."
        ),
    }
    return policies.get(
        structure_role,
        "Extract concrete patient findings, diagnoses, tests, results, and drugs stated in this block.",
    )


def build_entity_extraction_prompt(chunk: dict[str, Any]) -> str:
    structure_role = str(chunk.get("structure_role", "document"))
    payload = {
        "task": "Tag eligible medical spans in target_text.",
        "role_policy": _role_policy(structure_role),
        "rules": [
            'Return exactly {"mentions": [[quote, type], ...]}. Return {"mentions": []} when no span is eligible.',
            "Extract only from target_text. context_before and context_after are classification context and must never be quoted.",
            "Tag every eligible patient-specific occurrence. Scan target_text line by line and do not stop after the most important terms.",
            "structure_role is a hint, not a restriction. Tag visible eligible spans of every allowed type when a block contains mixed content.",
            "Return each eligible occurrence separately, including repeated text at different positions.",
            "For one occurrence, choose one best type. Never emit duplicate or alternative labels for the same span.",
            "Use the shortest complete span that preserves the clinical meaning, severity, location, visible result unit, or drug strength.",
            f"{TYPE_DRUG}: medication or therapeutic product. Include its visible strength and an immediate quantity such as x 2 ống or x 4 viên. Do not include infusion volume, route, frequency, infusion rate, or later administration instructions.",
            f"{TYPE_TEST_NAME}: assay, laboratory marker, imaging/procedure name, vital-sign label, or measured variable. Short labels such as BC, N, HBsAg, Anti HBe, Ure, Creatinin, GOT, GPT, GGT, ALT, and AST are valid.",
            f"{TYPE_TEST_RESULT}: only the value or qualitative finding produced by a test, measurement, imaging study, ECG, pathology, or examination.",
            f"Never label a drug strength, drug quantity, route, frequency, infusion rate, or medication instruction as {TYPE_TEST_RESULT}.",
            f"When a test and value are both visible, return separate {TYPE_TEST_NAME} and {TYPE_TEST_RESULT} spans. Keep a numeric result with its unit.",
            f"Keep one complete qualitative imaging, ECG, endoscopy, pathology, or biopsy finding as {TYPE_TEST_RESULT}.",
            f"{TYPE_DIAGNOSIS}: an explicitly stated diagnosis, condition, pathogen diagnosis, or item in a diagnosis/history/problem list. Never tag an entire narrative sentence as a diagnosis when only a smaller condition phrase is eligible.",
            f"{TYPE_SYMPTOM}: a patient complaint, sign, examination finding, mental/functional state, exposure, or normal/negated clinical observation.",
            "Classify from local context, not from the term alone. The same quote may have different types at different positions.",
            "Include negated and historical patient mentions; assertion labels are assigned later.",
            "Exclude demographics, dates, addresses, section headings, bare anatomy, generic words such as disease/problem/drug/test, and non-patient educational or hypothetical mentions.",
            "Do not return ICD-10/RxNorm candidates.",
            "If the text lists many independent medical terms in one sentence, split them into separate mentions unless the words form one single named condition, drug, test, or report finding.",
        ],
        "pattern_examples_not_target_text": [
            {
                "text": "GOT: 542 U/l",
                "mentions": [["GOT", TYPE_TEST_NAME], ["542 U/l", TYPE_TEST_RESULT]],
            },
            {
                "text": "Fortex 25mg x 4 viên, sáng 2 viên",
                "mentions": [["Fortex 25mg x 4 viên", TYPE_DRUG]],
            },
        ],
        "allowed_types": list(ALLOWED_TYPES),
        "target": {
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "structure_role": structure_role,
            "context_before": chunk.get("context_before", ""),
            "target_text": chunk["text"],
            "context_after": chunk.get("context_after", ""),
        },
        "response_schema": {
            "mentions": [
                ["exact quote from target_text", TYPE_SYMPTOM]
            ]
        },
        "mention_format": ["exact_quote", "type"],
    }
    return json.dumps(payload, ensure_ascii=False)
