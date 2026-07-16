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
    medication_match_score,
    medication_tty_score,
    normalize_prescription_text,
    strip_drug_count,
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
    ) -> None:
        self.records = records
        self.aliases = aliases
        self.diagnosis_lexical_index = DiagnosisLexicalIndex.build(records, aliases)
        self.medication_lexical_index = MedicationLexicalIndex.build(records)

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


class MedicationLexicalIndex:
    def __init__(self, postings: dict[str, tuple[CandidateRecord, ...]]) -> None:
        self.postings = postings

    @classmethod
    def build(cls, records: dict[tuple[str, str], CandidateRecord]) -> "MedicationLexicalIndex":
        postings: dict[str, list[CandidateRecord]] = defaultdict(list)
        for (concept_type, _), record in records.items():
            if concept_type != TYPE_DRUG or not record.name:
                continue
            ingredient = _medication_record_key(record.name)
            if len(ingredient) < 3:
                continue
            postings[ingredient].append(record)
        return cls({key: tuple(value) for key, value in postings.items()})

    @classmethod
    def empty(cls) -> "MedicationLexicalIndex":
        return cls({})

    def lookup(self, text: str, limit: int = 20) -> list[CandidateHit]:
        ingredient = medication_ingredient_key(text)
        if not ingredient:
            return []
        scored: list[CandidateHit] = []
        for record in self.postings.get(ingredient, ()):
            score = (
                0.70
                + medication_match_score(text, record.name)
                + medication_tty_score(text, record.ttys)
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
                    score = 0.45 + min(doc_score, 1.0) * 0.4 - min(record.priority, 100) * 0.0005
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
    return score


def _diagnosis_search_variants(text: str) -> list[str]:
    variants: list[tuple[str, str]] = []
    key = normalize_key(text)
    _add_variant(variants, "diagnosis_search", _diagnosis_search_key(key))
    _add_variant(variants, "diagnosis_core_search", _diagnosis_search_key(_strip_diagnosis_prefix(key)))
    _add_variant(variants, "diagnosis_spelling_search", _diagnosis_search_key(_normalize_diagnosis_spelling(key)))
    _add_variant(variants, "diagnosis_unspecified_search", _diagnosis_search_key(_strip_unspecified_suffix(key)))
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
    return _expand_diagnosis_shorthand(key)


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
