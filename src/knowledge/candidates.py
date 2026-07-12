"""Slim candidate index for ICD-10 and RxNorm lookup."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from core.config import CODED_TYPES, TYPE_DIAGNOSIS, TYPE_DRUG
from core.text import normalize_key


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    code: str
    name: str
    system: str
    concept_type: str
    priority: int = 100
    archive: bool = False
    ttys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateHit:
    record: CandidateRecord
    source: str
    score: float


class SlimCandidateIndex:
    def __init__(
        self,
        records: dict[tuple[str, str], CandidateRecord],
        aliases: dict[tuple[str, str], tuple[str, ...]],
    ) -> None:
        self.records = records
        self.aliases = aliases

    @classmethod
    def empty(cls) -> "SlimCandidateIndex":
        return cls({}, {})

    def lookup(self, text: str, concept_type: str, limit: int = 10) -> list[CandidateHit]:
        if concept_type not in CODED_TYPES:
            return []

        hits: list[CandidateHit] = []
        seen: set[str] = set()
        for source, key in _query_variants(text, concept_type):
            codes = self.aliases.get((concept_type, key), ())
            for rank, code in enumerate(codes):
                record = self.records.get((concept_type, code))
                if record is None or code in seen:
                    continue
                seen.add(code)
                score = 1.0 - min(rank, 20) * 0.01 - min(record.priority, 100) * 0.001
                hits.append(CandidateHit(record=record, source=source, score=score))
                if len(hits) >= limit:
                    return hits
        return hits


def load_slim_candidate_index(candidate_dir: Path) -> SlimCandidateIndex:
    if not candidate_dir.exists():
        return SlimCandidateIndex.empty()

    records: dict[tuple[str, str], CandidateRecord] = {}
    inline_aliases: dict[tuple[str, str], list[str]] = {}
    for file_name in ("icd10_candidates.jsonl", "rxnorm_candidates.jsonl"):
        path = candidate_dir / file_name
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            concept_type = str(row.get("type") or "")
            code = str(row.get("code") or "")
            if concept_type not in CODED_TYPES or not code:
                continue
            name = str(row.get("name") or row.get("name_vi") or row.get("name_en") or "")
            records[(concept_type, code)] = CandidateRecord(
                code=code,
                name=name,
                system=str(row.get("system") or ""),
                concept_type=concept_type,
                priority=_safe_int(row.get("priority"), 100),
                archive=bool(row.get("archive")),
                ttys=tuple(str(item) for item in row.get("ttys") or () if item),
            )
            for alias in _row_alias_norms(row):
                _add_alias_code(inline_aliases, concept_type, alias, code)

    aliases: dict[tuple[str, str], tuple[str, ...]] = {}
    alias_path = candidate_dir / "candidate_aliases.jsonl"
    if alias_path.exists():
        for row in _read_jsonl(alias_path):
            concept_type = str(row.get("type") or "")
            alias = str(row.get("alias_norm") or "")
            if concept_type not in CODED_TYPES or not alias:
                continue
            codes: list[str] = []
            for item in row.get("candidates") or []:
                if isinstance(item, dict):
                    code = str(item.get("code") or "")
                    if code and (concept_type, code) in records:
                        codes.append(code)
            if codes:
                aliases[(concept_type, alias)] = tuple(codes)

    for key, codes in inline_aliases.items():
        if key not in aliases:
            aliases[key] = tuple(_sort_codes(key[0], codes, records))

    return SlimCandidateIndex(records, aliases)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_alias_norms(row: dict[str, object]) -> list[str]:
    values: list[str] = []
    for key in ("alias_norms", "aliases"):
        raw = row.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
        elif isinstance(raw, str):
            values.append(raw)
    for key in ("name", "name_vi", "name_en"):
        value = str(row.get(key) or "")
        if value:
            values.append(normalize_key(value))

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        alias = normalize_key(value)
        if alias and alias not in seen:
            seen.add(alias)
            output.append(alias)
    return output


def _add_alias_code(
    aliases: dict[tuple[str, str], list[str]],
    concept_type: str,
    alias: str,
    code: str,
) -> None:
    key = (concept_type, alias)
    codes = aliases.setdefault(key, [])
    if code not in codes:
        codes.append(code)


def _sort_codes(
    concept_type: str,
    codes: list[str],
    records: dict[tuple[str, str], CandidateRecord],
) -> list[str]:
    return sorted(
        codes,
        key=lambda code: (
            records.get((concept_type, code), CandidateRecord(code, "", "", concept_type)).priority,
            len(code),
            code,
        ),
    )


def _query_variants(text: str, concept_type: str) -> list[tuple[str, str]]:
    key = normalize_key(text)
    variants: list[tuple[str, str]] = []
    _add_variant(variants, "exact", key)

    if concept_type == TYPE_DRUG:
        normalized_drug = _normalize_drug_units(_normalize_drug_spacing(key))
        _add_variant(variants, "drug_unit", normalized_drug)
        _add_variant(variants, "drug_no_count", _strip_drug_count(normalized_drug))
        _add_variant(variants, "drug_core", _strip_drug_route_frequency(normalized_drug))
        _add_variant(variants, "drug_no_release_token", _strip_drug_release_tokens(_strip_drug_route_frequency(normalized_drug)))
        _add_variant(variants, "drug_ingredient", _strip_drug_modifiers(normalized_drug))
    elif concept_type == TYPE_DIAGNOSIS:
        _add_variant(variants, "diagnosis_core", _strip_diagnosis_prefix(key))
        _add_variant(variants, "diagnosis_unspecified", _strip_unspecified_suffix(key))
        _add_variant(variants, "diagnosis_spelling", _normalize_diagnosis_spelling(key))
        _add_variant(variants, "diagnosis_with_benh_prefix", f"benh {key}")

    return variants


def _add_variant(variants: list[tuple[str, str]], source: str, key: str) -> None:
    key = " ".join((key or "").split())
    if key and all(existing != key for _, existing in variants):
        variants.append((source, key))


def _normalize_drug_units(key: str) -> str:
    key = re.sub(r"\bmg\s*/\s*ml\b", "mg/ml", key)
    key = re.sub(r"\bmcg\s*/\s*ml\b", "mcg/ml", key)
    return key


def _normalize_drug_spacing(key: str) -> str:
    key = re.sub(r"(\d)(mg|mcg|g|ml|iu|units?)\b", r"\1 \2", key)
    key = re.sub(r"\b(mg|mcg|g|ml|iu)\s*/\s*ml\b", r"\1/ml", key)
    key = re.sub(r"\bq\s*(\d+)\s*h\s*:?\s*prn\b", r"q\1h:prn", key)
    return " ".join(key.split())


def _strip_drug_count(key: str) -> str:
    key = re.sub(r"\s+x\s*\d+(?:\s+(?:ngay|day|days))?\b.*$", "", key)
    return " ".join(key.split())


def _strip_drug_route_frequency(key: str) -> str:
    tokens = key.split()
    stop = {
        "po",
        "iv",
        "im",
        "sc",
        "bid",
        "tid",
        "qid",
        "qam",
        "qhs",
        "daily",
        "prn",
        "x",
    }
    kept: list[str] = []
    for token in tokens:
        if token in stop or re.fullmatch(r"q\d+h:?prn?", token):
            break
        kept.append(token)
    return " ".join(kept)


def _strip_drug_release_tokens(key: str) -> str:
    tokens = [token for token in key.split() if token not in {"xl", "xr", "er", "sr", "cr"}]
    return " ".join(tokens)


def _strip_drug_modifiers(key: str) -> str:
    tokens = key.split()
    stop = {"mg", "mcg", "g", "ml", "iu", "unit", "units", "tablet", "capsule", "solution"}
    kept: list[str] = []
    for token in tokens:
        if token in stop or any(ch.isdigit() for ch in token):
            break
        kept.append(token)
    return " ".join(kept)


def _strip_diagnosis_prefix(key: str) -> str:
    prefixes = (
        "chan doan mac benh ",
        "chan doan ",
        "duoc chan doan mac benh ",
        "duoc chan doan ",
        "mac benh ",
        "benh ",
        "theo doi ",
        "nghi ngo ",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _strip_unspecified_suffix(key: str) -> str:
    suffixes = (
        " khong xac dinh",
        " khong dac hieu",
        " unspecified",
        " nos",
    )
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _normalize_diagnosis_spelling(key: str) -> str:
    replacements = {
        "sung huyet": "xung huyet",
        "tieu duong": "dai thao duong",
        "da day thuc quan": "da day - thuc quan",
        "tuyp": "type",
    }
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key
