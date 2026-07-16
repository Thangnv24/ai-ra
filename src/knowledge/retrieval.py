"""Candidate retrieval over the local ontology index."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.config import CODED_TYPES, TYPE_DIAGNOSIS, TYPE_DRUG
from core.text import normalize_key
from knowledge.candidates import SlimCandidateIndex
from knowledge.ontology import OntologyIndex


class CandidateRetriever:
    def __init__(self, index: OntologyIndex, slim_index: SlimCandidateIndex | None = None):
        self.index = index
        self.slim_index = slim_index or SlimCandidateIndex.empty()

    @lru_cache(maxsize=8192)
    def candidates_for(self, text: str, concept_type: str, limit: int = 5) -> tuple[str, ...]:
        if concept_type not in CODED_TYPES:
            return ()
        if concept_type == TYPE_DRUG:
            slim_hits = self.slim_index.lookup(text, concept_type, limit=max(limit * 6, 30))
            if slim_hits:
                return (slim_hits[0].record.code,)
        if concept_type == TYPE_DIAGNOSIS:
            slim_hits = self.slim_index.lookup(text, concept_type, limit=max(limit * 6, 30))
            selected = _select_diagnosis_codes(text, slim_hits, limit=limit)
            if selected or slim_hits:
                return selected
        codes: list[str] = []
        for entry in self.index.lookup(text, concept_type, limit=limit):
            codes.append(entry.code)
        for hit in self.slim_index.lookup(text, concept_type, limit=max(limit * 4, 20)):
            codes.append(hit.record.code)
        return tuple(_unique(codes)[:limit])

    def candidate_rows_for(self, text: str, concept_type: str, limit: int = 10) -> list[dict[str, Any]]:
        if concept_type not in CODED_TYPES:
            return []

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        if concept_type == TYPE_DRUG:
            for hit in self.slim_index.lookup(text, concept_type, limit=max(limit * 3, 20)):
                code = hit.record.code
                if code in seen:
                    continue
                seen.add(code)
                rows.append(
                    {
                        "code": code,
                        "name": hit.record.name,
                        "system": hit.record.system,
                        "source": hit.source,
                        "priority": hit.record.priority,
                        "score": round(hit.score, 6),
                        "archive": hit.record.archive,
                        "ttys": list(hit.record.ttys),
                    }
                )
                if len(rows) >= limit:
                    return rows
        for entry in self.index.lookup(text, concept_type, limit=limit):
            if entry.code in seen:
                continue
            seen.add(entry.code)
            rows.append(
                {
                    "code": entry.code,
                    "name": entry.name,
                    "system": entry.system,
                    "source": "ontology_seed",
                    "priority": entry.priority,
                }
            )
        for hit in self.slim_index.lookup(text, concept_type, limit=max(limit * 2, 10)):
            code = hit.record.code
            if code in seen:
                continue
            seen.add(code)
            rows.append(
                {
                    "code": code,
                    "name": hit.record.name,
                    "system": hit.record.system,
                    "source": hit.source,
                    "priority": hit.record.priority,
                    "score": round(hit.score, 6),
                    "archive": hit.record.archive,
                    "ttys": list(hit.record.ttys),
                }
            )
            if len(rows) >= limit:
                break
        return rows


def _unique(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            output.append(code)
    return output


def _select_diagnosis_codes(text: str, hits: list[Any], limit: int) -> tuple[str, ...]:
    key = normalize_key(text)
    if key in {"u tuyen", "khoi u truc trang", "benh ly chat trang", "hoi chung nao gan"}:
        return ()
    if not hits:
        return ()

    top = hits[0]
    if top.score < 0.70:
        return ()
    selected = [top.record.code]
    if len(hits) < 2 or limit < 2:
        return tuple(selected)

    second = hits[1]
    close_exact_siblings = (
        top.source in {"exact", "diagnosis_canonical"}
        and second.source in {"exact", "diagnosis_canonical"}
        and second.score >= 0.90
        and top.score - second.score <= 0.015
        and not any(marker in key for marker in ("khong xac dinh", "khong dac hieu", "unspecified"))
    )
    if close_exact_siblings:
        selected.append(second.record.code)
    return tuple(selected[:limit])
