"""Fast deterministic medical mention extraction."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from core.config import TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_NAME, TYPE_TEST_RESULT
from core.text import normalize_key, trim_span_text
from extraction.labs import extract_lab_spans


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    text: str
    type: str
    score: float = 1.0


SYMPTOM_TERMS = (
    "ho \u0111\u1eddm xanh",
    "t\u1ee9c ng\u1ef1c",
    "\u0111au th\u01b0\u1ee3ng v\u1ecb",
    "\u1ee3 h\u01a1i",
    "\u0111au nh\u1ee9c",
    "s\u1ed1t \u0111au",
    "t\u00e1o b\u00f3n",
    "lo \u00e2u",
    "m\u1ea5t ng\u1ee7",
    "kh\u00f3 th\u1edf",
    "\u0111au b\u1ee5ng",
    "\u0111au ng\u1ef1c",
    "\u0111au \u0111\u1ea7u",
    "ch\u00f3ng m\u1eb7t",
    "bu\u1ed3n n\u00f4n",
    "ti\u00eau ch\u1ea3y",
    "y\u1ebfu n\u1eeda ng\u01b0\u1eddi",
    "nh\u00ecn song th\u1ecb",
    "ti\u1ec3u ti\u1ec7n kh\u00f4ng t\u1ef1 ch\u1ee7",
    "lo\u00e9t \u0111au",
    "lo\u00e9t \u0111\u1ecf",
    "lo\u00e9t s\u01b0ng",
    "c\u01a1n co t\u1eed cung",
    "\u0111au v\u00f9ng h\u1ea1 s\u01b0\u1eddn ph\u1ea3i",
    "m\u1ec7t m\u1ecfi",
    "v\u00e0ng da",
    "ng\u1ee9a",
    "ch\u1ea3y d\u1ecbch",
    "s\u01b0ng n\u1ec1",
    "gi\u1ecdng kh\u00e0n",
    "n\u00f4n",
    "ho",
    "s\u1ed1t",
)

DIAGNOSIS_TERMS = (
    "b\u1ec7nh tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y - th\u1ef1c qu\u1ea3n",
    "tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y th\u1ef1c qu\u1ea3n",
    "benh trao nguoc da day - thuc quan",
    "trao nguoc da day thuc quan",
    "t\u0103ng huy\u1ebft \u00e1p",
    "cao huy\u1ebft \u00e1p",
    "tang huyet ap",
    "cao huyet ap",
    "\u0111\u00e1i th\u00e1o \u0111\u01b0\u1eddng type 2",
    "ti\u1ec3u \u0111\u01b0\u1eddng type 2",
    "dai thao duong type 2",
    "tieu duong type 2",
    "hen suy\u1ec5n",
    "hen suyen",
    "vi\u00eam ph\u1ed5i",
    "viem phoi",
    "x\u01a1 gan do r\u01b0\u1ee3u",
    "xo gan do ruou",
    "h\u1ed9i ch\u1ee9ng n\u00e3o gan",
    "hoi chung nao gan",
    "n\u1ed1t tuy\u1ebfn gi\u00e1p",
    "not tuyen giap",
    "t\u1eafc ngh\u1ebdn \u0111\u01b0\u1eddng m\u1eadt",
    "tac nghen duong mat",
    "vi\u00eam d\u1ea1 d\u00e0y",
    "viem da day",
    "ngh\u1ebdn t\u1eafc v\u00e0 h\u1eb9p \u0111\u1ed9ng m\u1ea1ch c\u1ea3nh",
    "h\u1eb9p \u0111\u1ed9ng m\u1ea1ch c\u1ea3nh",
    "xu\u1ea5t huy\u1ebft n\u1ed9i s\u1ecd kh\u00f4ng do ch\u1ea5n th\u01b0\u01a1ng",
    "b\u00e9o ph\u00ec",
    "t\u00e1ch th\u00e0nh \u0111\u1ed9ng m\u1ea1ch ch\u1ee7",
    "r\u00f2 \u0111\u1ed9ng - t\u0129nh m\u1ea1ch",
    "sa \u00e2m \u0111\u1ea1o",
    "b\u1ec7nh r\u1ec5 th\u1ea7n kinh tu\u1ef7 s\u1ed1ng",
    "b\u1ec7nh r\u1ec5 th\u1ea7n kinh t\u1ee7y s\u1ed1ng",
    "b\u1ec7nh graves",
    "t\u0103ng lipid m\u00e1u",
    "h\u1eb9p \u1ed1ng s\u1ed1ng",
    "lo\u00e9t b\u00e0n ch\u00e2n nhi\u1ec5m tr\u00f9ng",
    "thuy\u00ean t\u1eafc ph\u1ed5i",
    "nhi\u1ec5m tr\u00f9ng huy\u1ebft",
    "vi\u00eam t\u1ee7y x\u01b0\u01a1ng m\u00e3n t\u00ednh",
    "\u0111a u t\u1ee7y x\u01b0\u01a1ng",
    "n\u1ed1t s\u1ea7n tuy\u1ebfn gi\u00e1p",
    "s\u1ecfi m\u1eadt",
    "b\u1ec7nh ph\u1ed5i t\u1eafc ngh\u1ebdn m\u1ea1n t\u00ednh",
    "t\u00e2m th\u1ea7n ph\u00e2n li\u1ec7t",
    "thi\u1ebfu m\u00e1u",
    "kh\u1ed1i \u1edf ch\u1ed7 u\u1ed1n gan",
    "ung th\u01b0 bi\u1ec3u m\u00f4 tuy\u1ebfn gi\u00e1p nh\u00fa",
    "ung th\u01b0 bi\u1ec3u m\u00f4 tuy\u1ebfn gi\u1eadt nh\u00fa",
    "ung th\u01b0 bi\u1ec3u m\u00f4 tuy\u1ebfn",
    "ung th\u01b0 bi\u1ec3u m\u00f4 t\u1ebf b\u00e0o th\u1eadn",
    "gi\u00e3n \u0111\u01b0\u1eddng d\u1eabn m\u1eadt",
    "h\u1eb9p \u1ed1ng m\u1eadt ch\u1ee7",
    "u \u00e1c c\u1ee7a \u0111\u1ea7u tu\u1ef5",
    "u \u00e1c c\u1ee7a \u0111\u1ea7u t\u1ee5y",
    "kh\u1ed1i u c\u00f3 ngu\u1ed3n g\u1ed1c t\u1eeb \u0111\u01b0\u1eddng m\u1eadt t\u1ee5y",
    "kh\u1ed1i u c\u00f3 ngu\u1ed3n g\u1ed1c t\u1eeb \u0111\u01b0\u1eddng m\u1eadt tu\u1ef5",
    "b\u00e0n ch\u00e2n v\u1eb9o b\u1ea9m sinh",
    "g\u00e3y x\u01b0\u01a1ng s\u01b0\u1eddn",
    "\u0111\u1ee5ng d\u1eadp ph\u1ed5i",
    "v\u1ebft th\u01b0\u01a1ng th\u1ea5u b\u1ee5ng",
    "d\u00e0y ni\u00eam m\u1ea1c xoang h\u00e0m",
    "b\u1ec7nh \u0111\u1ed9ng m\u1ea1ch v\u00e0nh",
    "suy tim",
    "b\u1ec7nh m\u1ea1ch m\u00e1u ngo\u1ea1i bi\u00ean",
    "ng\u01b0ng th\u1edf khi ng\u1ee7 do t\u1eafc ngh\u1ebdn",
    "ung th\u01b0 bi\u1ec3u m\u00f4 t\u1ebf b\u00e0o v\u1ea3y",
    "kh\u1ed1i u th\u1ea7n kinh n\u1ed9i ti\u1ebft",
    "u n\u1ed9i ti\u1ebft",
    "r\u00f2 \u1ed1ng tu\u1ef5 m\u1eadt",
    "r\u00f2 \u1ed1ng t\u1ee5y m\u1eadt",
    "nhi\u1ec5m clostridioides difficile",
    "nhi\u1ec5m tr\u00f9ng v\u1ebft m\u1ed5",
    "t\u1ed5n th\u01b0\u01a1ng d\u00e2y thanh qu\u1ea3n",
    "hypertension",
    "diabetes mellitus",
    "asthma",
    "pneumonia",
    "gerd",
)

TEST_TERMS = (
    "t\u1ed5ng ph\u00e2n t\u00edch t\u1ebf b\u00e0o m\u00e1u",
    "c\u00f4ng th\u1ee9c m\u00e1u",
    "x\u00e9t nghi\u1ec7m m\u00e1u",
    "wbc",
    "twbc",
    "neut%",
    "lyph%",
    "lymph%",
    "rbc",
    "hgb",
    "plt",
    "crp",
    "glucose",
    "creatinine",
)

DRUG_BASE_TERMS = (
    "chlorpheniramine",
    "capsaicin",
    "metoprolol succinate",
    "docusate sodium",
    "acetaminophen",
    "paracetamol",
    "amlodipine",
    "aspirin",
    "guaifenesin",
    "nystatin",
    "pravastatin",
    "senna",
    "clonazepam",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "ibuprofen",
    "amoxicillin",
    "octreotide",
    "flagyl",
    "metronidazole",
    "lasix",
    "furosemide",
)

LAB_PAIR_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9%+\-/]*(?:\s*\([^)]+\))?)\s*[:=]\s*"
    r"(?P<result>[<>]?\s*\d+(?:[,.]\d+)?(?:\s*(?:mg/dL|g/L|mmol/L|mmol|IU/L|U/L|%))?)",
    re.IGNORECASE,
)

DIAG_CONTEXT_RE = re.compile(
    r"(?:ch\u1ea9n\s*\u0111o\u00e1n|chan\s*doan|diagnosis|dx)\s*"
    r"(?::|m\u1eafc\s+b\u1ec7nh|mac\s+benh|m\u1eafc|mac|l\u00e0|la)\s*"
    r"(?P<diag>[^.;\n]+)",
    re.IGNORECASE,
)


class MedicalNER:
    def __init__(self, lexicon_paths: tuple[Path, ...] = ()) -> None:
        extra = _load_external_lexicons(lexicon_paths)
        self._test_terms = TEST_TERMS + tuple(extra.get(TYPE_TEST_NAME, ()))
        self._symptom_patterns = _compile_terms(SYMPTOM_TERMS + tuple(extra.get(TYPE_SYMPTOM, ())))
        self._diagnosis_patterns = _compile_terms(DIAGNOSIS_TERMS + tuple(extra.get(TYPE_DIAGNOSIS, ())))
        self._test_patterns = _compile_terms(self._test_terms)
        self._drug_patterns = _compile_terms(DRUG_BASE_TERMS + tuple(extra.get(TYPE_DRUG, ())))

    def extract(self, text: str) -> list[SpanCandidate]:
        spans: list[SpanCandidate] = []
        spans.extend(self._extract_lab_pairs(text))
        spans.extend(self._extract_known_lab_values(text))
        spans.extend(self._extract_drugs(text))
        spans.extend(self._extract_context_diagnoses(text))
        spans.extend(_extract_phrase_matches(text, self._diagnosis_patterns, TYPE_DIAGNOSIS, 0.95))
        spans.extend(_extract_phrase_matches(text, self._symptom_patterns, TYPE_SYMPTOM, 0.8))
        spans.extend(_extract_phrase_matches(text, self._test_patterns, TYPE_TEST_NAME, 0.7))
        return _resolve_overlaps(spans)

    def _extract_lab_pairs(self, text: str) -> list[SpanCandidate]:
        return [
            SpanCandidate(span.start, span.end, span.text, span.type, 0.99)
            for span in extract_lab_spans(text)
        ]

    def _extract_known_lab_values(self, text: str) -> list[SpanCandidate]:
        spans: list[SpanCandidate] = []
        value = r"(?P<result>[<>]?\s*\d+(?:[,.]\d+)?(?:\s*(?:mg/dL|g/L|mmol/L|mmol|IU/L|U/L|%))?)"
        for term in sorted(set(self._test_terms), key=len, reverse=True):
            if len(term) < 3:
                continue
            pattern = re.compile(rf"(?<!\w)(?P<name>{re.escape(term)})(?:\s*\([^)]+\))?\s*(?:l\u00e0|:|=)\s*{value}", re.IGNORECASE)
            for match in pattern.finditer(text):
                name_start, name_end = match.span("name")
                result_start, result_end = match.span("result")
                name_start, name_end, name_text = trim_span_text(text, name_start, name_end)
                result_start, result_end, result_text = trim_span_text(text, result_start, result_end)
                spans.append(SpanCandidate(name_start, name_end, name_text, TYPE_TEST_NAME, 0.99))
                spans.append(SpanCandidate(result_start, result_end, result_text, TYPE_TEST_RESULT, 0.99))
        return spans

    def _extract_drugs(self, text: str) -> list[SpanCandidate]:
        spans: list[SpanCandidate] = []
        for pattern in self._drug_patterns:
            for match in pattern.finditer(text):
                start = match.start()
                end = _extend_drug_end(text, match.end())
                start, end, span_text = trim_span_text(text, start, end)
                if span_text:
                    spans.append(SpanCandidate(start, end, span_text, TYPE_DRUG, 0.98))
        return spans

    def _extract_context_diagnoses(self, text: str) -> list[SpanCandidate]:
        spans: list[SpanCandidate] = []
        for match in DIAG_CONTEXT_RE.finditer(text):
            start, end = match.span("diag")
            raw = match.group("diag")
            for prefix in ("m\u1eafc ", "mac "):
                if normalize_key(raw).startswith(normalize_key(prefix)):
                    start += len(prefix)
                    break
            start, end, span_text = trim_span_text(text, start, end)
            if span_text and len(span_text) >= 4 and not _bad_context_diagnosis(span_text):
                spans.append(SpanCandidate(start, end, span_text, TYPE_DIAGNOSIS, 0.9))
        return spans


def _compile_terms(terms: tuple[str, ...]) -> list[re.Pattern[str]]:
    ordered = sorted(set(terms), key=len, reverse=True)
    patterns: list[re.Pattern[str]] = []
    for term in ordered:
        escaped = re.escape(term)
        pattern = rf"(?<!\w){escaped}(?!\w)"
        patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns


def _load_external_lexicons(paths: tuple[Path, ...]) -> dict[str, list[str]]:
    terms: dict[str, list[str]] = {
        TYPE_SYMPTOM: [],
        TYPE_DIAGNOSIS: [],
        TYPE_TEST_NAME: [],
        TYPE_DRUG: [],
    }
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                term = (row.get("term") or "").strip()
                concept_type = (row.get("type") or "").strip()
                if term and concept_type in terms:
                    terms[concept_type].append(term)
    return terms


def _extract_phrase_matches(
    text: str,
    patterns: list[re.Pattern[str]],
    concept_type: str,
    score: float,
) -> list[SpanCandidate]:
    spans: list[SpanCandidate] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start, end, span_text = trim_span_text(text, match.start(), match.end())
            if span_text:
                spans.append(SpanCandidate(start, end, span_text, concept_type, score))
    return spans


def _extend_drug_end(text: str, offset: int) -> int:
    end = offset
    max_end = min(len(text), offset + 80)
    terminators = [
        "\n",
        ";",
        ",",
        " \u0111i\u1ec1u tr\u1ecb ",
        " dieu tri ",
        " for ",
    ]
    next_stop = max_end
    lower_tail = text[offset:max_end].casefold()
    for token in terminators:
        idx = lower_tail.find(token)
        if idx >= 0:
            next_stop = min(next_stop, offset + idx)
    candidate = text[offset:next_stop]
    # Keep common dose/route/frequency tokens after the drug name.
    match = re.match(
        r"(?:\s+(?:xl|oral|suspension|tablet|capsule|solution|"
        r"\d+(?:[.,-]\d+)?(?:mg|mcg|g|ml)?|mg/ml|mcg/ml|mg/dl|mg|mcg|g|ml|"
        r"po|iv|im|sc|bid|tid|qid|qam|qhs|q\d+h:?prn|daily|prn|x|\+))+",
        candidate,
        flags=re.IGNORECASE,
    )
    if match:
        end = offset + match.end()
    return max(end, offset)


def _looks_like_lab_name(name: str) -> bool:
    key = normalize_key(name)
    if key in {normalize_key(term) for term in TEST_TERMS}:
        return True
    return bool(re.search(r"[A-Za-z%]", name)) and len(name) <= 80


def _bad_context_diagnosis(text: str) -> bool:
    key = normalize_key(text)
    if len(key.split()) <= 2 and key in {"khac", "tham do", "va dieu tri", "hinh anh"}:
        return True
    bad_prefixes = (
        "hinh anh",
        "chup x quang",
        "sieu am",
        "ct ",
        "mri ",
        "dien tam do",
        "ecg",
        "/ tham do",
        "va dieu tri",
    )
    return any(key.startswith(prefix) for prefix in bad_prefixes)


def _resolve_overlaps(spans: list[SpanCandidate]) -> list[SpanCandidate]:
    priority = {
        TYPE_DRUG: 5,
        TYPE_DIAGNOSIS: 4,
        TYPE_TEST_RESULT: 3,
        TYPE_TEST_NAME: 2,
        TYPE_SYMPTOM: 1,
    }
    ordered = sorted(
        spans,
        key=lambda s: (-(s.end - s.start), -priority.get(s.type, 0), -s.score, s.start),
    )
    selected: list[SpanCandidate] = []
    for span in ordered:
        if span.start >= span.end:
            continue
        if any(_overlap(span, existing) for existing in selected):
            continue
        selected.append(span)
    return sorted(selected, key=lambda s: (s.start, s.end))


def _overlap(left: SpanCandidate, right: SpanCandidate) -> bool:
    return left.start < right.end and right.start < left.end
