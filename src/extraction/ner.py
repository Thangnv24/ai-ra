"""Fast deterministic medical mention extraction."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from core.config import TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_NAME, TYPE_TEST_RESULT
from core.medication import extend_medication_span_end
from core.text import normalize_key, trim_span_text
from extraction.labs import extract_lab_spans


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    text: str
    type: str
    score: float = 1.0
    source: str = "rule"


SYMPTOM_TERMS = (
    "\u0111\u00e1nh tr\u1ed1ng ng\u1ef1c",
    "c\u1ea3m gi\u00e1c th\u1eaft ch\u1eb7t ng\u1ef1c",
    "gi\u1ea3m dung n\u1ea1p g\u1eafng s\u1ee9c",
    "ngh\u1eb9t ng\u1ef1c",
    "n\u00f4n ra m\u00e1u",
    "kh\u00f3 th\u1edf khi g\u1eafng s\u1ee9c",
    "kh\u00f3 th\u1edf khi n\u1eb1m",
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
    "ng\u1ea5t x\u1ec9u",
    "m\u1ea5t \u00fd th\u1ee9c",
    "qu\u1ea7ng s\u00e1ng",
    "kh\u00f3 th\u1edf v\u1ec1 \u0111\u00eam",
    "ph\u00f9 m\u1eaft c\u00e1 ch\u00e2n",
    "\u0111au ch\u00e2n",
    "\u0111au l\u01b0ng",
    "ch\u01b0\u1edbng b\u1ee5ng",
    "l\u00fa l\u1eabn",
    "kh\u00f3 kh\u0103n khi nh\u00ecn g\u1ea7n",
    "kh\u00f3 kh\u0103n v\u1ec1 th\u1ecb l\u1ef1c g\u1ea7n",
    "r\u1ed1i lo\u1ea1n th\u1ecb gi\u00e1c",
    "\u1ea3o gi\u00e1c th\u1ecb gi\u00e1c",
    "\u1ea3o gi\u00e1c th\u00ednh gi\u00e1c",
    "t\u1ef1 t\u1eed",
    "kh\u00f3 nu\u1ed1t",
    "kh\u00e0n ti\u1ebfng",
    "lower abdominal pain",
    "ch\u1ea3y m\u00e1u nhi\u1ec1u",
    "\u0111\u1eddm",
    "\u0111au b\u1ee5ng qu\u1eb7n",
    "\u0111i ngo\u00e0i ra m\u00e1u",
)

DIAGNOSIS_TERMS = (
    "b\u1ec7nh tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y - th\u1ef1c qu\u1ea3n",
    "tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y th\u1ef1c qu\u1ea3n",
    "benh trao nguoc da day - thuc quan",
    "trao nguoc da day thuc quan",
    "vi\u00eam d\u1ea1 d\u00e0y ru\u1ed9t do virus",
    "vi\u00eam d\u1ea1 d\u00e0y - ru\u1ed9t do virus",
    "vi\u00eam d\u1ea1 d\u00e0y ru\u1ed9t",
    "vi\u00eam d\u1ea1 d\u00e0y - ru\u1ed9t",
    "viral gastroenteritis",
    "t\u0103ng huy\u1ebft \u00e1p",
    "cao huy\u1ebft \u00e1p",
    "tang huyet ap",
    "cao huyet ap",
    "\u0111\u00e1i th\u00e1o \u0111\u01b0\u1eddng type 2",
    "\u0111\u00e1i th\u00e1o \u0111\u01b0\u1eddng",
    "ti\u1ec3u \u0111\u01b0\u1eddng type 2",
    "ti\u1ec3u \u0111\u01b0\u1eddng",
    "dai thao duong type 2",
    "dai thao duong",
    "tieu duong type 2",
    "tieu duong",
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
    "vi\u00eam tuy\u1ebfn m\u1ed3 h\u00f4i",
    "ngo\u1ea1i t\u00e2m thu nh\u0129",
    "ngo\u1ea1i t\u00e2m thu th\u1ea5t",
    "b\u1ec7nh tim m\u1ea1ch do x\u01a1 v\u1eefa \u0111\u1ed9ng m\u1ea1ch",
    "ph\u00ecnh \u0111\u1ed9ng m\u1ea1ch ch\u1ee7",
    "rung nh\u0129 k\u00e8m \u0111\u00e1p \u1ee9ng th\u1ea5t nhanh",
    "nh\u1ed3i m\u00e1u c\u01a1 tim v\u00f9ng d\u01b0\u1edbi",
    "nh\u1ed3i m\u00e1u c\u01a1 tim v\u00f9ng d\u01b0\u1edbi c\u0169",
    "tim to",
    "s\u1ecfi \u0111o\u1ea1n cu\u1ed1i \u1ed1ng m\u1eadt ch\u1ee7",
    "s\u1ecfi \u1ed1ng d\u1eabn m\u1eadt chung \u0111o\u1ea1n cu\u1ed1i",
    "b\u1ec7nh l\u00fd ch\u1ea5t tr\u1eafng",
    "b\u1ec7nh \u0111a x\u01a1 c\u1ee9ng",
    "\u1ea3o gi\u00e1c do r\u01b0\u1ee3u",
    "lo\u1ea1n th\u1ea7n",
    "n\u1ed1t tuy\u1ebfn gi\u00e1p th\u00f9y tr\u00e1i",
    "u c\u01a1 tr\u01a1n t\u1eed cung",
    "u \u00e1c tr\u1ef1c tr\u00e0ng",
    "kh\u1ed1i u tr\u1ef1c tr\u00e0ng",
    "u tuy\u1ebfn",
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
    "monitor holter",
    "si\u00eau \u00e2m tim qua th\u00e0nh ng\u1ef1c",
    "ch\u1ee5p x-quang ng\u1ef1c",
    "x-quang ng\u1ef1c",
    "ph\u00e2n t\u00edch n\u01b0\u1edbc ti\u1ec3u",
    "b\u1ea3n ph\u00e2n t\u00edch n\u01b0\u1edbc ti\u1ec3u",
    "\u0111i\u1ec7n t\u00e2m \u0111\u1ed3",
    "ecg",
    "d\u1ea5u hi\u1ec7u sinh t\u1ed3n",
    "b\u1ea3ng c\u00f4ng th\u1ee9c sinh h\u00f3a m\u00e1u c\u01a1 b\u1ea3n",
    "b\u1ea3ng c\u00f4ng th\u1ee9c m\u00e1u",
    "b\u1ea3ng ch\u1ee9c n\u0103ng gan",
    "troponin",
    "ch\u1ee5p c\u1eaft l\u1edbp vi t\u00ednh s\u1ecd n\u00e3o",
    "n\u1ed9i soi th\u1ef1c qu\u1ea3n - d\u1ea1 d\u00e0y - t\u00e1 tr\u00e0ng",
    "ch\u1ee5p c\u1ed9ng h\u01b0\u1edfng t\u1eeb m\u1eadt t\u1ee5y",
    "x\u00e9t nghi\u1ec7m ch\u1ee9c n\u0103ng gan",
    "si\u00eau \u00e2m gan m\u1eadt",
    "si\u00eau \u00e2m doppler \u0111\u1ed9ng m\u1ea1ch",
    "ch\u1ee5p ct s\u1ecd",
    "ch\u1ecdc d\u00f2 d\u1ecbch n\u00e3o t\u1ee7y",
    "b\u0103ng nh\u00f3m oligoclonal",
    "ch\u1ecdc h\u00fat b\u1eb1ng kim nh\u1ecf",
    "x\u00e9t nghi\u1ec7m t\u1ebf b\u00e0o h\u1ecdc",
    "huy\u1ebft c\u1ea7u t\u1ed1",
    "si\u00eau \u00e2m t\u1eed cung",
    "cea",
    "mri v\u00f9ng ch\u1eadu",
    "sinh thi\u1ebft",
    "n\u1ed9i soi m\u1eadt t\u1ee5y ng\u01b0\u1ee3c d\u00f2ng",
)

RESULT_TERMS = (
    "kh\u00f4ng ghi nh\u1eadn g\u00ec b\u1ea5t th\u01b0\u1eddng",
    "kh\u00f4ng c\u00f3 g\u00ec \u0111\u00e1ng ch\u00fa \u00fd",
    "nh\u1ecbp xoang chi\u1ebfm \u01b0u th\u1ebf",
    "nh\u1ecbp xoang",
    "men gan t\u0103ng",
    "v\u1eadn t\u1ed1c d\u00f2ng ch\u1ea3y t\u0103ng r\u00f5",
    "t\u0103ng nh\u1eb9",
    "\u00e2m t\u00ednh",
    "d\u01b0\u01a1ng t\u00ednh",
    "b\u00ecnh th\u01b0\u1eddng",
    "b\u1ea5t th\u01b0\u1eddng",
)

DRUG_BASE_TERMS = (
    "chlorpheniramine",
    "capsaicin",
    "metoprolol succinate",
    "metoprolol",
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
    "doxycycline",
    "atenolol",
    "omeprazole",
    "propofol",
    "phentolamine",
    "levophed",
    "norepinephrine",
    "vancomycin",
    "ceftazidime",
    "zosyn",
    "bactrim",
    "cipro",
    "ciprofloxacin",
    "seroquel",
    "quetiapine",
    "tylenol",
    "mucinex",
    "lorazepam",
    "ativan",
    "lovenox",
    "enoxaparin",
    "clopidogrel",
    "ranexa",
    "ranolazine",
    "heparin",
    "nitroglycerin",
    "azithromycin",
    "iron",
    "nsaid",
    "nsaids",
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

_EXPLICIT_DIAGNOSIS_RE = re.compile(
    r"(?:duoc\s+)?chan\s*doan(?:\s+so\s+bo)?(?:\s+(?:la|mac|cho))?\s*:?\s*$"
)
_RESULT_HEADINGS = (
    "ket qua xet nghiem",
    "ket qua phong thi nghiem",
    "ket qua laboratory",
    "ket qua chan doan hinh anh",
    "ket qua chup anh",
    "ket qua hinh anh",
    "ket qua chup",
    "cac ket qua chan doan khac",
    "chan doan hinh anh va tham do",
)
_DIAGNOSIS_HEADINGS = (
    "cac phat hien chan doan khac",
    "chan doan so bo",
    "chan doan:",
)
_RESULT_TEST_CUES = (
    "xet nghiem",
    "phong thi nghiem",
    "laboratory",
    "sieu am",
    "x quang",
    "x-quang",
    "chup ",
    "ct ",
    "mri",
    "noi soi",
    "sinh thiet",
    "choc hut",
    "fna",
    "te bao hoc",
    "dien tam do",
    "ecg",
    "ekg",
    "holter",
    "xa hinh",
    "thong tim",
)
_RESULT_RELATION_CUES = (
    "cho thay",
    "ghi nhan",
    "phat hien",
    "am tinh",
    "duong tinh",
    "phu hop voi",
    "goi y",
)
_CONTEXT_HEADING_RESETS = (
    "trieu chung",
    "benh su",
    "dien bien benh",
    "dau hieu lam sang",
    "ket qua kham",
    "thu tuc",
    "thu thuat",
    "dieu tri",
)


class MedicalNER:
    def __init__(self, lexicon_paths: tuple[Path, ...] = ()) -> None:
        extra = _load_external_lexicons(lexicon_paths)
        self._test_terms = TEST_TERMS + tuple(extra.get(TYPE_TEST_NAME, ()))
        self._symptom_patterns = _compile_terms(SYMPTOM_TERMS + tuple(extra.get(TYPE_SYMPTOM, ())))
        self._diagnosis_patterns = _compile_terms(DIAGNOSIS_TERMS + tuple(extra.get(TYPE_DIAGNOSIS, ())))
        self._test_patterns = _compile_terms(self._test_terms)
        self._result_patterns = _compile_terms(RESULT_TERMS)
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
        spans.extend(self._extract_contextual_results(text))
        return _resolve_overlaps(spans)

    def _extract_contextual_results(self, text: str) -> list[SpanCandidate]:
        spans: list[SpanCandidate] = []
        for pattern in self._result_patterns:
            for match in pattern.finditer(text):
                start, end, span_text = trim_span_text(text, match.start(), match.end())
                if not span_text:
                    continue
                key = normalize_key(span_text)
                context = normalize_key(text[max(0, start - 140) : min(len(text), end + 60)])
                if key.startswith("nhip xoang") or _looks_like_result_context(context):
                    spans.append(SpanCandidate(start, end, span_text, TYPE_TEST_RESULT, 0.92))
        return spans

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
    return extend_medication_span_end(text, offset)


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


def resolve_span_types(text: str, spans: list[SpanCandidate]) -> list[SpanCandidate]:
    """Override uncertain LLM types only when deterministic evidence is strong."""

    resolved: list[SpanCandidate] = []
    for span in spans:
        key = normalize_key(span.text)
        concept_type = _deterministic_type(key, text, span.start, span.end, span.type) or span.type
        resolved.append(replace(span, type=concept_type))
    return resolved


def _deterministic_type(
    key: str,
    text: str,
    start: int,
    end: int,
    proposed_type: str,
) -> str | None:
    if not key:
        return None
    if _starts_with_term(key, DRUG_BASE_TERMS):
        return TYPE_DRUG
    if key in _normalized_terms(TEST_TERMS):
        return TYPE_TEST_NAME
    if _looks_like_explicit_diagnosis_context(text, start):
        if proposed_type in {TYPE_DIAGNOSIS, TYPE_SYMPTOM, TYPE_TEST_RESULT} or key in _normalized_terms(
            DIAGNOSIS_TERMS + SYMPTOM_TERMS
        ):
            return TYPE_DIAGNOSIS
    if _looks_like_observation_result_context(text, start, end):
        if proposed_type in {TYPE_DIAGNOSIS, TYPE_TEST_RESULT} or key in _normalized_terms(DIAGNOSIS_TERMS):
            return TYPE_TEST_RESULT
    if key in _normalized_terms(RESULT_TERMS):
        context = normalize_key(text[max(0, start - 140) : min(len(text), end + 60)])
        if key.startswith("nhip xoang") or _looks_like_result_context(context):
            return TYPE_TEST_RESULT
    if key in _normalized_terms(DIAGNOSIS_TERMS):
        return TYPE_DIAGNOSIS
    if key in _normalized_terms(SYMPTOM_TERMS):
        return TYPE_SYMPTOM
    if re.fullmatch(r"[<>]?\s*\d+(?:[,.]\d+)?(?:\s*[a-z%/^0-9]+)?", key):
        context = normalize_key(text[max(0, start - 120) : start])
        if _looks_like_result_context(context):
            return TYPE_TEST_RESULT
    return None


def _looks_like_explicit_diagnosis_context(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    prefix = normalize_key(text[line_start:start])
    if any(heading in prefix for heading in _RESULT_HEADINGS):
        return False
    return bool(_EXPLICIT_DIAGNOSIS_RE.search(prefix))


def _looks_like_observation_result_context(text: str, start: int, end: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)

    raw_prefix = text[line_start:start]
    line_key = normalize_key(text[line_start:line_end])
    prefix = normalize_key(raw_prefix)
    suffix = normalize_key(text[end : min(line_end, end + 100)])
    if any(heading in line_key for heading in _DIAGNOSIS_HEADINGS) and not any(
        heading in line_key for heading in _RESULT_HEADINGS
    ):
        return False
    if raw_prefix.rfind("(") > raw_prefix.rfind(")"):
        return False

    last_test_cue = _last_context_cue_start(prefix, _RESULT_TEST_CUES)
    last_relation_cue = _last_context_cue_start(prefix, _RESULT_RELATION_CUES)
    prefix_has_test = last_test_cue >= 0
    prefix_has_relation = last_relation_cue >= last_test_cue >= 0
    parenthetical_test = suffix.lstrip().startswith("(") and _contains_context_cue(suffix, _RESULT_TEST_CUES)
    test_assignment = prefix_has_test and prefix.rstrip().endswith(":")
    if (prefix_has_test and prefix_has_relation) or parenthetical_test or test_assignment:
        return True
    if any(prefix.endswith(marker) for marker in ("cua", "doi voi", "tren")):
        return False
    if any(suffix.startswith(cue) for cue in _RESULT_RELATION_CUES):
        return False
    if any(prefix.startswith(marker) for marker in _CONTEXT_HEADING_RESETS):
        return False
    return _nearest_context_heading(text, line_start) == TYPE_TEST_RESULT


def _contains_context_cue(value: str, cues: tuple[str, ...]) -> bool:
    return _last_context_cue_start(value, cues) >= 0


def _last_context_cue_start(value: str, cues: tuple[str, ...]) -> int:
    starts = [
        match.start()
        for cue in cues
        if cue.strip()
        for match in re.finditer(rf"(?<!\w){re.escape(cue.strip())}(?!\w)", value)
    ]
    return max(starts, default=-1)


def _nearest_context_heading(text: str, line_start: int) -> str | None:
    previous_lines = text[:line_start].splitlines()[-8:]
    for raw_line in reversed(previous_lines):
        key = normalize_key(raw_line.strip(" -*\t"))
        if not key:
            continue
        if any(heading in key for heading in _DIAGNOSIS_HEADINGS) and not any(
            heading in key for heading in _RESULT_HEADINGS
        ):
            return TYPE_DIAGNOSIS
        if any(heading in key for heading in _RESULT_HEADINGS):
            return TYPE_TEST_RESULT
        if any(key.startswith(marker) for marker in _CONTEXT_HEADING_RESETS):
            return None
    return None


@lru_cache(maxsize=16)
def _normalized_terms(terms: tuple[str, ...]) -> frozenset[str]:
    return frozenset(normalize_key(term) for term in terms)


def _starts_with_term(key: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        term_key = normalize_key(term)
        if key == term_key or key.startswith(term_key + " "):
            return True
        if key.startswith(term_key) and len(key) > len(term_key) and key[len(term_key)].isdigit():
            return True
    return False


def _looks_like_result_context(context: str) -> bool:
    cues = (
        "ket qua",
        "xet nghiem",
        "chan doan hinh anh",
        "cho thay",
        "ghi nhan",
        "dien tam do",
        "ecg",
        "holter",
        "sieu am",
        "x quang",
        "ct ",
        "mri",
        "sinh thiet",
        "choc hut",
        "choc do",
    )
    return any(cue in context for cue in cues)
