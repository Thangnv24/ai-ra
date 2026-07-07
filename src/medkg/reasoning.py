"""Lightweight internal relation inference.

The public phase-1 JSON format does not expose a relation field. These helpers
are kept for future scoring variants and for debugging, but the pipeline does
not emit their output.
"""

from __future__ import annotations

from dataclasses import dataclass

from medkg.config import TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_NAME, TYPE_TEST_RESULT
from medkg.schema import Concept


@dataclass(frozen=True)
class Relation:
    source_index: int
    target_index: int
    relation_type: str


def infer_relations(concepts: list[Concept]) -> list[Relation]:
    relations: list[Relation] = []
    for i, left in enumerate(concepts):
        for j, right in enumerate(concepts):
            if i == j:
                continue
            distance = abs(left.position[0] - right.position[0])
            if distance > 120:
                continue
            if left.type == TYPE_DRUG and right.type in {TYPE_SYMPTOM, TYPE_DIAGNOSIS}:
                relations.append(Relation(i, j, "drug_related_to_condition"))
            elif left.type == TYPE_TEST_NAME and right.type == TYPE_TEST_RESULT:
                relations.append(Relation(i, j, "test_has_result"))
    return relations

