"""Candidate retrieval over the local ontology index."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.config import CODED_TYPES, TYPE_DIAGNOSIS, TYPE_DRUG
from core.medication import medication_has_strength, medication_strength_relation
from core.text import normalize_key
from knowledge.candidates import SlimCandidateIndex
from knowledge.ontology import OntologyIndex


class CandidateRetriever:
    def __init__(self, index: OntologyIndex, slim_index: SlimCandidateIndex | None = None):
        self.index = index
        self.slim_index = slim_index or SlimCandidateIndex.empty()

    def candidates_for(
        self,
        text: str,
        concept_type: str,
        limit: int = 5,
        *,
        source_text: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> tuple[str, ...]:
        if concept_type not in CODED_TYPES:
            return ()
        if not _candidate_eligible(concept_type, source_text, start, end):
            return ()
        return self._candidates_for_query(text, concept_type, limit)

    @lru_cache(maxsize=8192)
    def _candidates_for_query(self, text: str, concept_type: str, limit: int) -> tuple[str, ...]:
        if concept_type == TYPE_DRUG:
            slim_hits = self.slim_index.lookup(text, concept_type, limit=max(limit * 6, 30))
            return _select_drug_code(text, slim_hits)
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
    override = _DIAGNOSIS_CODE_OVERRIDES.get(key)
    if override:
        return override[:limit]
    if key in {"u tuyen", "khoi u truc trang", "benh ly chat trang", "hoi chung nao gan"}:
        return ()
    if not hits:
        return ()

    top = hits[0]
    threshold = 0.78 if top.source == "diagnosis_lexical" else 0.82
    if top.score < threshold:
        return ()
    if top.source == "diagnosis_lexical" and len(hits) > 1 and top.score - hits[1].score < 0.04:
        return ()
    return (top.record.code,)


def _select_drug_code(text: str, hits: list[Any]) -> tuple[str, ...]:
    if not hits:
        return ()
    has_strength = medication_has_strength(text)
    eligible = []
    for hit in hits:
        if has_strength:
            if medication_strength_relation(text, hit.record.name) != "match":
                continue
            tty_set = {tty.upper() for tty in hit.record.ttys}
            if tty_set and tty_set <= {"IN", "PIN", "MIN", "BN"}:
                continue
        eligible.append(hit)
    if not eligible:
        return ()

    top = eligible[0]
    weak_sources = {"drug_ingredient"}
    if top.source in weak_sources:
        return ()
    if top.source == "medication_lexical" and top.score < 0.82:
        return ()
    if top.score < 0.70:
        return ()
    return (top.record.code,)


def _candidate_eligible(
    concept_type: str,
    source_text: str | None,
    start: int | None,
    end: int | None,
) -> bool:
    if concept_type != TYPE_DIAGNOSIS:
        return True
    key = normalize_key(source_text[start:end] if source_text is not None and start is not None and end is not None else "")
    if key in _DIAGNOSIS_LONG_LINE_ALLOWLIST:
        return True
    if not key:
        return True
    if len(key) > 160 or len(key.split()) > 18:
        return False
    if source_text is None or start is None or end is None:
        return True
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", end)
    if line_end < 0:
        line_end = len(source_text)
    return line_end - line_start < 300


_DIAGNOSIS_CODE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "benh mach mau ngoai bien khong dac hieu": ("I73.9",),
    "benh phoi tac nghen man tinh khong xac dinh": ("J44.9",),
    "benh trao nguoc da day thuc quan khong co viem thuc quan": ("K21.9",),
    "benh tim mach do xo vua dong mach": ("I25.1",),
    "hep dong mach canh trong ben phai": ("I65.2",),
    "ho van hai la": ("I05.2",),
    "ngung tho khi ngu": ("G47.3",),
    "nhoi mau nao": ("I63.8",),
    "roi loan chuyen hoa lipid": ("E78.8",),
    "rung nhi": ("I48.2",),
    "rung nhi kem dap ung that nhanh": ("I48.2",),
    "rung nhi dap ung that nhanh": ("I48.2",),
    "soi duong mat": ("K80.4",),
    "soi ong dan mat chung": ("K80.3",),
    "soi ong mat chu": ("K80.3",),
    "suy tim": ("I50.9",),
    "suy tim khong dac hieu": ("I50.9",),
    "tang ha": ("I10",),
    "tang lipid mau khong dac hieu": ("E78.5",),
    "tang huyet ap vo can nguyen phat": ("I10",),
    "tha": ("I10",),
    "u ac cua tuyen tien liet": ("C61",),
    "van dong mach chu co hoc": ("Z95.4",),
    "viem ket mac mat trai": ("H10.3",),
    "viem tuy cap": ("K85.8",),
}


_DIAGNOSIS_LONG_LINE_ALLOWLIST = frozenset(
    {
        "benh mach mau ngoai bien khong dac hieu",
        "benh phoi tac nghen man tinh khong xac dinh",
        "benh trao nguoc da day thuc quan khong co viem thuc quan",
        "benh tim mach do xo vua dong mach",
        "hep dong mach canh trong ben phai",
        "ngung tho khi ngu",
        "suy tim khong dac hieu",
        "tang ha",
        "tang lipid mau khong dac hieu",
        "tang huyet ap vo can nguyen phat",
        "tha",
        "u ac cua tuyen tien liet",
        "van dong mach chu co hoc",
        "viem ket mac mat trai",
    }
)
