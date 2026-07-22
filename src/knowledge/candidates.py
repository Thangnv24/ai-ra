"""Slim candidate index for ICD-10 and RxNorm lookup."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from core.config import CODED_TYPES, TYPE_DIAGNOSIS, TYPE_DRUG
from core.medication import (
    medication_ingredient_key,
    medication_lookup_keys,
    medication_match_score,
    medication_tty_score,
    normalize_prescription_text,
    strip_drug_count,
    strip_drug_context,
    strip_drug_modifiers,
    strip_drug_release_tokens,
    strip_drug_route_frequency,
)
from core.text import normalize_key, similarity


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
        curated_aliases: dict[tuple[str, str], tuple[str, ...]] | None = None,
        curated_profile_candidates: dict[tuple[str, str, str], tuple[str, ...]] | None = None,
    ) -> None:
        self.records = records
        self.aliases = aliases
        self.curated_aliases = curated_aliases or {}
        self.curated_profile_candidates = curated_profile_candidates or {}
        self.diagnosis_lexical_index = DiagnosisLexicalIndex.build(records, aliases)
        self.medication_lexical_index = MedicationLexicalIndex.build(records, aliases)

    @classmethod
    def empty(cls) -> "SlimCandidateIndex":
        return cls({}, {})

    def lookup(self, text: str, concept_type: str, limit: int = 10) -> list[CandidateHit]:
        if concept_type not in CODED_TYPES:
            return []

        hits: list[CandidateHit] = []
        seen: set[str] = set()
        exact_key = normalize_key(text)
        for rank, code in enumerate(self.curated_aliases.get((concept_type, exact_key), ())):
            record = self.records.get((concept_type, code))
            if record is None or code in seen:
                continue
            seen.add(code)
            hits.append(
                CandidateHit(
                    record=record,
                    source="curated_exact",
                    score=1.65 - min(rank, 20) * 0.025,
                )
            )
        for source, key in _query_variants(text, concept_type):
            codes = self.aliases.get((concept_type, key), ())
            for rank, code in enumerate(codes):
                record = self.records.get((concept_type, code))
                if record is None or code in seen:
                    continue
                seen.add(code)
                score = _hit_score(text, concept_type, source, rank, record)
                hits.append(CandidateHit(record=record, source=source, score=score))
        if concept_type == TYPE_DIAGNOSIS and len(hits) < limit:
            for rank, hit in enumerate(self.diagnosis_lexical_index.lookup(text, limit=max(limit * 4, 20))):
                code = hit.record.code
                if code in seen:
                    continue
                seen.add(code)
                hits.append(
                    CandidateHit(
                        record=hit.record,
                        source=hit.source,
                        score=hit.score - min(rank, 20) * 0.005,
                    )
                )
        elif concept_type == TYPE_DRUG:
            for hit in self.medication_lexical_index.lookup(text, limit=max(limit * 4, 30)):
                code = hit.record.code
                if code in seen:
                    continue
                seen.add(code)
                hits.append(hit)
        hits.sort(key=lambda hit: (-hit.score, hit.record.priority, hit.record.code))
        return hits[:limit]

    def candidates_for_profile(
        self,
        text: str,
        concept_type: str,
        source_text: str | None,
        start: int | None,
        end: int | None,
    ) -> tuple[str, ...] | None:
        profile = _candidate_line_profile(source_text, start, end)
        return self.curated_profile_candidates.get(
            (concept_type, normalize_key(text), profile),
        )


class MedicationLexicalIndex:
    def __init__(
        self,
        postings: dict[str, tuple[CandidateRecord, ...]],
        alias_support: dict[str, int] | None = None,
    ) -> None:
        self.postings = postings
        self.alias_support = alias_support or {}

    @classmethod
    def build(
        cls,
        records: dict[tuple[str, str], CandidateRecord],
        aliases: dict[tuple[str, str], tuple[str, ...]] | None = None,
    ) -> "MedicationLexicalIndex":
        postings: dict[str, list[CandidateRecord]] = defaultdict(list)
        for (concept_type, _), record in records.items():
            if concept_type != TYPE_DRUG or not record.name:
                continue
            for lookup_key in _medication_record_keys(record.name):
                if len(lookup_key) >= 3:
                    postings[lookup_key].append(record)

        alias_support: Counter[str] = Counter()
        for (concept_type, _), codes in (aliases or {}).items():
            if concept_type == TYPE_DRUG:
                alias_support.update(codes)
        return cls(
            {key: tuple(value) for key, value in postings.items()},
            dict(alias_support),
        )

    @classmethod
    def empty(cls) -> "MedicationLexicalIndex":
        return cls({})

    def lookup(self, text: str, limit: int = 20) -> list[CandidateHit]:
        lookup_keys = medication_lookup_keys(text)
        if not lookup_keys:
            return []
        scored: list[CandidateHit] = []
        records: dict[str, CandidateRecord] = {}
        for lookup_key in lookup_keys:
            for record in self.postings.get(lookup_key, ()):
                records.setdefault(record.code, record)
        for record in records.values():
            score = (
                0.70
                + medication_match_score(text, record.name)
                + medication_tty_score(text, record.ttys)
                + min(self.alias_support.get(record.code, 0), 10) * 0.002
                - min(record.priority, 100) * 0.001
            )
            scored.append(CandidateHit(record=record, source="medication_lexical", score=score))
        scored.sort(key=lambda hit: (-hit.score, hit.record.priority, hit.record.code))
        return scored[:limit]


def _medication_record_key(name: str) -> str:
    """Fast build-time equivalent of medication_ingredient_key for RxNorm names."""

    key = normalize_key(name)
    match = re.search(
        r"\s+(?:\d|oral\b|tablet\b|capsule\b|solution\b|suspension\b|cream\b|"
        r"ointment\b|injection\b|patch\b|spray\b|powder\b|extended\b|release\b)",
        key,
    )
    if match:
        key = key[: match.start()]
    key = re.sub(r"\s*\[[^\]]+\]\s*$", "", key)
    return " ".join(key.split())


def _medication_record_keys(name: str) -> tuple[str, ...]:
    keys = [_medication_record_key(name)]
    for brand in re.findall(r"\[([^\]]+)\]", name):
        brand_key = medication_ingredient_key(brand)
        if len(brand_key) >= 3 and brand_key not in keys:
            keys.append(brand_key)
    return tuple(key for key in keys if key)


@dataclass(frozen=True, slots=True)
class DiagnosisAliasDoc:
    alias: str
    codes: tuple[str, ...]
    tokens: frozenset[str]
    grams: frozenset[str]


class DiagnosisLexicalIndex:
    def __init__(
        self,
        records: dict[tuple[str, str], CandidateRecord],
        docs: tuple[DiagnosisAliasDoc, ...],
        token_index: dict[str, tuple[int, ...]],
        gram_index: dict[str, tuple[int, ...]],
    ) -> None:
        self.records = records
        self.docs = docs
        self.token_index = token_index
        self.gram_index = gram_index
        self.doc_count = len(docs)

    @classmethod
    def build(
        cls,
        records: dict[tuple[str, str], CandidateRecord],
        aliases: dict[tuple[str, str], tuple[str, ...]],
    ) -> "DiagnosisLexicalIndex":
        docs: list[DiagnosisAliasDoc] = []
        token_postings: dict[str, list[int]] = defaultdict(list)
        gram_postings: dict[str, list[int]] = defaultdict(list)
        for (concept_type, alias), codes in aliases.items():
            if concept_type != TYPE_DIAGNOSIS:
                continue
            alias_key = _diagnosis_search_key(alias)
            tokens = frozenset(_diagnosis_tokens(alias_key))
            grams = frozenset(_char_ngrams(alias_key))
            if not tokens and not grams:
                continue
            valid_codes = tuple(
                code
                for code in codes
                if (TYPE_DIAGNOSIS, code) in records
            )
            if not valid_codes:
                continue
            doc_id = len(docs)
            docs.append(DiagnosisAliasDoc(alias=alias_key, codes=valid_codes, tokens=tokens, grams=grams))
            for token in tokens:
                token_postings[token].append(doc_id)
            for gram in grams:
                gram_postings[gram].append(doc_id)
        return cls(
            records,
            tuple(docs),
            {key: tuple(value) for key, value in token_postings.items()},
            {key: tuple(value) for key, value in gram_postings.items()},
        )

    @classmethod
    def empty(cls) -> "DiagnosisLexicalIndex":
        return cls({}, (), {}, {})

    def lookup(self, text: str, limit: int = 20) -> list[CandidateHit]:
        if not self.docs:
            return []

        best_by_code: dict[str, CandidateHit] = {}
        for query in _diagnosis_search_variants(text):
            for doc_score, doc in self._rank_docs(query, limit=max(limit * 4, 40)):
                if doc_score < _diagnosis_score_threshold(query):
                    continue
                for code in doc.codes:
                    record = self.records.get((TYPE_DIAGNOSIS, code))
                    if record is None:
                        continue
                    score = (
                        0.45
                        + min(doc_score, 1.0) * 0.4
                        + diagnosis_qualifier_adjustment(text, record.name)
                        - min(record.priority, 100) * 0.0005
                    )
                    current = best_by_code.get(code)
                    if current is None or score > current.score:
                        best_by_code[code] = CandidateHit(
                            record=record,
                            source="diagnosis_lexical",
                            score=score,
                        )
        hits = sorted(best_by_code.values(), key=lambda hit: (-hit.score, hit.record.priority, hit.record.code))
        return hits[:limit]

    def _rank_docs(self, query: str, limit: int) -> list[tuple[float, DiagnosisAliasDoc]]:
        tokens = frozenset(_diagnosis_tokens(query))
        grams = frozenset(_char_ngrams(query))
        if len(tokens) < 2 and len(normalize_key(query)) < 7:
            return []

        preliminary: Counter[int] = Counter()
        for token in tokens:
            postings = self.token_index.get(token, ())
            if not postings or len(postings) > 1600:
                continue
            weight = 1.0 + math.log((self.doc_count + 1) / (len(postings) + 1))
            for doc_id in postings:
                preliminary[doc_id] += weight
        for gram in grams:
            postings = self.gram_index.get(gram, ())
            if not postings or len(postings) > 2200:
                continue
            weight = 0.18 + 0.08 * math.log((self.doc_count + 1) / (len(postings) + 1))
            for doc_id in postings:
                preliminary[doc_id] += weight

        if not preliminary:
            return []

        scored: list[tuple[float, DiagnosisAliasDoc]] = []
        for doc_id, _ in preliminary.most_common(240):
            doc = self.docs[doc_id]
            score = _diagnosis_similarity_score(query, tokens, grams, doc)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda row: (-row[0], min(self.records[(TYPE_DIAGNOSIS, code)].priority for code in row[1].codes), row[1].codes[0]))
        return scored[:limit]


def load_slim_candidate_index(
    candidate_dir: Path,
    *,
    min_training_file_support: int = 1,
) -> SlimCandidateIndex:
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

    curated_aliases: dict[tuple[str, str], tuple[str, ...]] = {}
    curated_profile_candidates: dict[tuple[str, str, str], tuple[str, ...]] = {}
    training_path = candidate_dir / "candidate_training_lexicon.json"
    training_payload: dict[str, object] = {}
    if training_path.exists():
        raw_payload = json.loads(training_path.read_text(encoding="utf-8-sig"))
        if isinstance(raw_payload, dict):
            training_payload = raw_payload
        for row in training_payload.get("supplemental_records") or []:
            if not isinstance(row, dict):
                continue
            if min_training_file_support > 1:
                continue
            concept_type = str(row.get("type") or "")
            code = str(row.get("code") or "").strip()
            if concept_type not in CODED_TYPES or not code or (concept_type, code) in records:
                continue
            records[(concept_type, code)] = CandidateRecord(
                code=code,
                name=str(row.get("name") or ""),
                system=str(row.get("system") or ""),
                concept_type=concept_type,
                priority=_safe_int(row.get("priority"), 20),
                archive=bool(row.get("archive")),
                ttys=tuple(str(item) for item in row.get("ttys") or () if item),
            )

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

    for row in training_payload.get("aliases") or []:
        if not isinstance(row, dict):
            continue
        concept_type = str(row.get("type") or "")
        alias = normalize_key(str(row.get("alias_norm") or ""))
        if concept_type not in CODED_TYPES or not alias:
            continue
        codes = tuple(
            str(item.get("code") or "").strip()
            for item in row.get("candidates") or []
            if isinstance(item, dict)
            and _safe_int(item.get("file_support"), 0) >= min_training_file_support
            and str(item.get("code") or "").strip()
            and (concept_type, str(item.get("code") or "").strip()) in records
        )
        if not codes:
            continue
        key = (concept_type, alias)
        curated_aliases[key] = codes
        aliases[key] = tuple(_unique_codes([*codes, *aliases.get(key, ())]))

        for profile in row.get("profiles") or []:
            if not isinstance(profile, dict):
                continue
            profile_name = str(profile.get("name") or "")
            if profile_name not in {"short_line", "long_line"}:
                continue
            if _safe_int(profile.get("file_support"), 0) < min_training_file_support:
                continue
            raw_preferred = profile.get("preferred_candidates")
            if not isinstance(raw_preferred, list):
                continue
            preferred = tuple(
                str(code).strip()
                for code in raw_preferred
                if str(code).strip() and (concept_type, str(code).strip()) in records
            )
            curated_profile_candidates[(concept_type, alias, profile_name)] = preferred

    return SlimCandidateIndex(records, aliases, curated_aliases, curated_profile_candidates)


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


def _unique_codes(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            output.append(code)
    return output


def _candidate_line_profile(
    source_text: str | None,
    start: int | None,
    end: int | None,
) -> str:
    if source_text is None or start is None or end is None:
        return "short_line"
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", end)
    if line_end < 0:
        line_end = len(source_text)
    return "short_line" if line_end - line_start < 300 else "long_line"


def _query_variants(text: str, concept_type: str) -> list[tuple[str, str]]:
    key = normalize_key(text)
    variants: list[tuple[str, str]] = []
    if concept_type == TYPE_DIAGNOSIS:
        canonical = _canonical_diagnosis_alias(key)
        if canonical:
            _add_variant(variants, "diagnosis_canonical", canonical)
    _add_variant(variants, "exact", key)

    if concept_type == TYPE_DRUG:
        normalized_drug = normalize_prescription_text(text)
        _add_variant(variants, "drug_unit", normalized_drug)
        _add_variant(variants, "drug_no_count", strip_drug_count(normalized_drug))
        drug_core = strip_drug_route_frequency(normalized_drug)
        _add_variant(variants, "drug_core", drug_core)
        _add_variant(variants, "drug_no_release_token", strip_drug_release_tokens(drug_core))
        _add_variant(variants, "drug_ingredient", strip_drug_modifiers(normalized_drug))
        contextual_core = strip_drug_context(normalized_drug)
        _add_variant(variants, "drug_context_core", contextual_core)
        _add_variant(variants, "drug_context_ingredient", strip_drug_modifiers(contextual_core))
    elif concept_type == TYPE_DIAGNOSIS:
        _add_variant(variants, "diagnosis_core", _strip_diagnosis_prefix(key))
        _add_variant(variants, "diagnosis_spelling", _normalize_diagnosis_spelling(key))
        _add_variant(variants, "diagnosis_unspecified", _strip_unspecified_suffix(key))
        _add_variant(variants, "diagnosis_chronic_shorthand", _expand_diagnosis_shorthand(key))
        _add_variant(variants, "diagnosis_with_benh_prefix", f"benh {key}")

    return variants


def _hit_score(
    text: str,
    concept_type: str,
    source: str,
    rank: int,
    record: CandidateRecord,
) -> float:
    source_penalty = {
        "exact": 0.0,
        "drug_unit": 0.005,
        "drug_no_count": 0.015,
        "drug_core": 0.03,
        "drug_no_release_token": 0.04,
        "drug_ingredient": 0.12,
        "drug_context_core": 0.035,
        "drug_context_ingredient": 0.10,
        "diagnosis_core": 0.02,
        "diagnosis_canonical": -0.03,
        "diagnosis_unspecified": 0.03,
        "diagnosis_spelling": 0.04,
        "diagnosis_chronic_shorthand": 0.045,
        "diagnosis_with_benh_prefix": 0.05,
    }.get(source, 0.08)
    score = 1.0 - source_penalty - min(rank, 20) * 0.01 - min(record.priority, 100) * 0.001
    if concept_type == TYPE_DRUG:
        score += medication_match_score(text, record.name)
        score += medication_tty_score(text, record.ttys)
    elif concept_type == TYPE_DIAGNOSIS:
        score += diagnosis_qualifier_adjustment(text, record.name)
    return score


def diagnosis_qualifier_adjustment(query_text: str, candidate_text: str) -> float:
    """Reward matching code-defining qualifiers and penalize conflicts."""

    query = _normalize_diagnosis_spelling(normalize_key(query_text))
    candidate = _normalize_diagnosis_spelling(normalize_key(candidate_text))
    score = 0.0

    query_stage = _diagnosis_stage(query)
    candidate_stage = _diagnosis_stage(candidate)
    score += _qualifier_pair_score(query_stage, candidate_stage, 0.16, -0.25, -0.45)

    query_acuity = _diagnosis_acuity(query)
    candidate_acuity = _diagnosis_acuity(candidate)
    score += _qualifier_pair_score(query_acuity, candidate_acuity, 0.08, -0.16, -0.30)

    query_side = _diagnosis_laterality(query)
    candidate_side = _diagnosis_laterality(candidate)
    score += _qualifier_pair_score(query_side, candidate_side, 0.10, -0.18, -0.35)

    query_type = _diagnosis_type(query)
    candidate_type = _diagnosis_type(candidate)
    score += _qualifier_pair_score(query_type, candidate_type, 0.10, -0.20, -0.35)

    query_unspecified = _has_unspecified_qualifier(query)
    candidate_unspecified = _has_unspecified_qualifier(candidate)
    if query_unspecified:
        score += 0.08 if candidate_unspecified else -0.10
    elif candidate_unspecified and any((query_stage, query_acuity, query_side, query_type)):
        score -= 0.08
    return score


def _qualifier_pair_score(
    query_value: str,
    candidate_value: str,
    match_bonus: float,
    missing_penalty: float,
    conflict_penalty: float,
) -> float:
    if not query_value:
        return 0.0
    if not candidate_value:
        return missing_penalty
    return match_bonus if query_value == candidate_value else conflict_penalty


def _diagnosis_stage(key: str) -> str:
    match = re.search(r"\bgiai doan\s+(cuoi|[1-5]|i{1,3}|iv|v)\b", key)
    if not match:
        return ""
    value = match.group(1)
    return {"cuoi": "5", "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}.get(value, value)


def _diagnosis_acuity(key: str) -> str:
    if re.search(r"\bcap(?: tinh)?\b", key):
        return "acute"
    if re.search(r"\bman(?: tinh)?\b", key):
        return "chronic"
    return ""


def _diagnosis_laterality(key: str) -> str:
    if re.search(r"\b(hai ben|bilateral)\b", key):
        return "bilateral"
    if re.search(r"\b(trai|left)\b", key):
        return "left"
    if re.search(r"\b(phai|right)\b", key):
        return "right"
    return ""


def _diagnosis_type(key: str) -> str:
    match = re.search(r"\b(?:type|tip)\s*([12])\b", key)
    return match.group(1) if match else ""


def _has_unspecified_qualifier(key: str) -> bool:
    return bool(re.search(r"\b(khong xac dinh|khong dac hieu|unspecified|nos)\b", key))


def _diagnosis_search_variants(text: str) -> list[str]:
    variants: list[tuple[str, str]] = []
    key = normalize_key(text)
    _add_variant(variants, "diagnosis_search", _diagnosis_search_key(key))
    _add_variant(variants, "diagnosis_core_search", _diagnosis_search_key(_strip_diagnosis_prefix(key)))
    _add_variant(variants, "diagnosis_spelling_search", _diagnosis_search_key(_normalize_diagnosis_spelling(key)))
    _add_variant(variants, "diagnosis_unspecified_search", _diagnosis_search_key(_strip_unspecified_suffix(key)))
    _add_variant(variants, "diagnosis_measurement_search", _diagnosis_search_key(_strip_diagnosis_measurements(key)))
    _add_variant(variants, "diagnosis_context_search", _diagnosis_search_key(_strip_diagnosis_context(key)))
    return [value for _, value in variants]


def _diagnosis_search_key(text: str) -> str:
    key = normalize_key(text)
    key = _normalize_diagnosis_spelling(key)
    key = _strip_diagnosis_prefix(key)
    key = _strip_unspecified_suffix(key)
    prefixes = (
        "tien su ",
        "co tien su ",
        "duoc phat hien ",
        "phat hien ",
        "dang dieu tri ",
        "da dieu tri ",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return " ".join(key.split())


def _diagnosis_tokens(key: str) -> list[str]:
    return [
        token
        for token in normalize_key(key).split()
        if len(token) >= 2 and token not in _DIAGNOSIS_STOP_TOKENS
    ]


def _char_ngrams(key: str, size: int = 3) -> list[str]:
    compact = "".join(ch for ch in normalize_key(key) if ch.isalnum())
    if len(compact) < size:
        return []
    return [compact[index : index + size] for index in range(len(compact) - size + 1)]


def _diagnosis_similarity_score(
    query: str,
    query_tokens: frozenset[str],
    query_grams: frozenset[str],
    doc: DiagnosisAliasDoc,
) -> float:
    if not query:
        return 0.0
    token_overlap = len(query_tokens & doc.tokens)
    gram_overlap = len(query_grams & doc.grams)
    if token_overlap == 0 and gram_overlap < 3:
        return 0.0

    token_recall = token_overlap / max(1, len(query_tokens))
    token_precision = token_overlap / max(1, len(doc.tokens))
    token_jaccard = token_overlap / max(1, len(query_tokens | doc.tokens))
    gram_jaccard = gram_overlap / max(1, len(query_grams | doc.grams))
    sequence_score = similarity(query, doc.alias)
    substring_bonus = 0.0
    if query in doc.alias or doc.alias in query:
        substring_bonus = 0.1
    elif " ".join(query.split()) in doc.alias:
        substring_bonus = 0.05
    if substring_bonus == 0.0 and token_recall < 0.55:
        return 0.0

    query_len = max(1, len(query))
    doc_len = max(1, len(doc.alias))
    length_ratio = min(query_len, doc_len) / max(query_len, doc_len)
    score = (
        0.34 * token_recall
        + 0.18 * token_precision
        + 0.16 * token_jaccard
        + 0.18 * gram_jaccard
        + 0.14 * sequence_score
        + substring_bonus
    )
    score -= max(0.0, 0.55 - length_ratio) * 0.12
    return score


def _diagnosis_score_threshold(query: str) -> float:
    tokens = _diagnosis_tokens(query)
    if len(tokens) <= 2:
        return 0.58
    if len(tokens) <= 4:
        return 0.5
    return 0.46


def _add_variant(variants: list[tuple[str, str]], source: str, key: str) -> None:
    key = " ".join((key or "").split())
    if key and all(existing != key for _, existing in variants):
        variants.append((source, key))


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
        "khong dac hieu": "khong xac dinh",
        "tuyp": "type",
    }
    for source, target in replacements.items():
        key = key.replace(source, target)
    token_replacements = {
        "dtd": "dai thao duong",
        "gut": "gout",
        "vkdt": "viem khop dang thap",
    }
    for source, target in token_replacements.items():
        key = re.sub(rf"\b{re.escape(source)}\b", target, key)
    key = re.sub(r"\btang\s+ha\b", "tang huyet ap", key)
    key = re.sub(r"\btang\s+huyet+t\s+ap\b", "tang huyet ap", key)
    key = re.sub(r"\btyp\s*ii\b", "type 2", key)
    return _expand_diagnosis_shorthand(key)


def _strip_diagnosis_measurements(key: str) -> str:
    key = normalize_key(key)
    key = re.sub(r"\b\d+(?:[\.,-]\d+)?\s*%\b", " ", key)
    key = re.sub(r"\b(?:m?rs|abcd2|balthazar)\s*\w*\b", " ", key)
    key = re.sub(r"\b\d+\s*(?:d|diem)\b", " ", key)
    return " ".join(key.split())


def _strip_diagnosis_context(key: str) -> str:
    key = normalize_key(key)
    key = re.sub(r"^(?:theo doi|nghi ngo|tien su|co tien su)\s+", "", key)
    key = re.sub(r"\s+(?:gio thu|cach day)\s+\d+\b.*$", "", key)
    return " ".join(key.split())


def _expand_diagnosis_shorthand(key: str) -> str:
    tokens = normalize_key(key).split()
    expanded: list[str] = []
    for index, token in enumerate(tokens):
        expanded.append(token)
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token in {"man", "cap"} and next_token != "tinh":
            expanded.append("tinh")
    return " ".join(expanded)


def _canonical_diagnosis_alias(key: str) -> str:
    normalized = _normalize_diagnosis_spelling(_strip_diagnosis_prefix(normalize_key(key)))
    exact_aliases = {
        "tang huyet ap": "benh tang huyet ap vo can nguyen phat",
        "phinh dong mach chu": "phinh dong mach chu vi tri khong xac dinh khong vo",
        "phinh dong mach chu nho": "phinh dong mach chu vi tri khong xac dinh khong vo",
        "viem da day": "viem da day khong xac dinh",
        "soi doan cuoi ong mat chu": "soi mat khong viem duong dan mat hay viem tui mat",
        "soi ong mat chu": "soi mat khong viem duong dan mat hay viem tui mat",
        "soi ong dan mat chung doan cuoi": "soi mat khong viem duong dan mat hay viem tui mat",
        "rung nhi": "rung nhi va hoac cuong nhi khong xac dinh",
        "rung nhi dap ung that nhanh": "rung nhi va hoac cuong nhi khong xac dinh",
        "rung nhi kem dap ung that nhanh": "rung nhi va hoac cuong nhi khong xac dinh",
        "nhoi mau co tim vung duoi cu": "nhoi mau co tim cu",
        "not tuyen giap": "buou giap don nhan khong doc",
        "not tuyen giap trai": "buou giap don nhan khong doc",
        "not tuyen giap thuy trai": "buou giap don nhan khong doc",
        "u co tron tu cung": "u co tron tu cung khong xac dinh",
        "u co tron tu cung khong xac dinh": "u co tron tu cung khong xac dinh",
        "tang lipid mau": "tang lipid mau khong xac dinh",
        "ao giac do ruou": "roi loan tam than va hoac hanh vi do su dung ruou loan than",
        "benh trao nguoc da day thuc quan khong co viem thuc quan": (
            "benh trao nguoc da day thuc quan khong co viem thuc quan"
        ),
    }
    return exact_aliases.get(normalized, "")


_DIAGNOSIS_STOP_TOKENS = {
    "benh",
    "chan",
    "doan",
    "mac",
    "duoc",
    "co",
    "hoi",
    "chung",
    "khac",
    "theo",
    "doi",
    "nghi",
    "ngo",
    "cua",
    "do",
    "va",
    "voi",
    "tai",
    "o",
    "the",
    "of",
    "and",
    "or",
    "in",
    "on",
    "to",
    "due",
    "with",
    "without",
    "other",
    "specified",
    "unspecified",
    "disease",
    "disorder",
    "syndrome",
}
