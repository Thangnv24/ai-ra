"""Medication text helpers for prescription-style drug mentions."""

from __future__ import annotations

import re

from core.text import normalize_key


ROUTE_TOKENS = {
    "po",
    "iv",
    "im",
    "sc",
    "sq",
    "sl",
    "pr",
    "topical",
    "oral",
}
FREQUENCY_TOKENS = {
    "ac",
    "pc",
    "bid",
    "bd",
    "bds",
    "tid",
    "tds",
    "qid",
    "qds",
    "qd",
    "qod",
    "qam",
    "qpm",
    "qhs",
    "qhour",
    "qh",
    "hs",
    "daily",
    "prn",
    "stat",
}
FORM_TOKENS = {
    "tablet",
    "tablets",
    "tab",
    "tabs",
    "capsule",
    "capsules",
    "cap",
    "caps",
    "solution",
    "suspension",
    "cream",
    "ointment",
    "oral",
    "injection",
    "patch",
    "spray",
    "drop",
    "drops",
    "extended",
    "release",
    "xl",
    "xr",
    "er",
    "sr",
    "cr",
}
COUNT_TOKENS = {"x", "times", "day", "days", "ngay"}
UNIT_TOKENS = {"mg/ml", "mcg/ml", "mg", "mcg", "g", "ml", "iu", "unit", "units", "%"}

_UNIT_PATTERN = r"mg/ml|mcg/ml|mg|mcg|g|ml|iu|unit|units|%"
_STRENGTH_RE = re.compile(rf"\b\d+(?:[\.,]\d+)?\s*(?:{_UNIT_PATTERN})\b")
_Q_INTERVAL_RE = re.compile(r"\bq\s*(\d+)\s*h\s*:?\s*prn\b")
_Q_INTERVAL_PLAIN_RE = re.compile(r"\bq\s*(\d+)\s*h\b")
_ABBREVIATION_REPLACEMENTS = (
    (re.compile(r"\bp\.?\s*o\.?\b"), "po"),
    (re.compile(r"\bp\.?\s*r\.?\b"), "pr"),
    (re.compile(r"\bp\.?\s*r\.?\s*n\.?\b"), "prn"),
    (re.compile(r"\bq\.?\s*a\.?\s*m\.?\b"), "qam"),
    (re.compile(r"\bq\.?\s*p\.?\s*m\.?\b"), "qpm"),
    (re.compile(r"\bq\.?\s*h\.?\s*s\.?\b"), "qhs"),
    (re.compile(r"\bb\.?\s*i\.?\s*d\.?\b"), "bid"),
    (re.compile(r"\bt\.?\s*i\.?\s*d\.?\b"), "tid"),
    (re.compile(r"\bq\.?\s*i\.?\s*d\.?\b"), "qid"),
    (re.compile(r"\bq\.?\s*d\.?\b"), "qd"),
)
_TAIL_TOKEN_RE = re.compile(
    rf"""
    (?:
        \s+
        (?:
            \d+(?:[\.,-]\d+)?\s*(?:{_UNIT_PATTERN})?
            | { _UNIT_PATTERN }
            | q\s*\d+\s*h\s*:?\s*prn
            | q\s*\d+\s*h
            | qam\s*:?\s*prn
            | qpm\s*:?\s*prn
            | qhs\s*:?\s*prn
            | bid\s*:?\s*prn
            | tid\s*:?\s*prn
            | qid\s*:?\s*prn
            | [A-Za-z]{{1,6}}\.[A-Za-z.]+
            | [A-Za-z][A-Za-z0-9/-]*
            | \+
            | x\s*\d+
        )
    )+
    """,
    re.IGNORECASE | re.VERBOSE,
)
_TAIL_ALLOWED_RE = re.compile(
    rf"""
    ^(?:
        \d+(?:[\.,-]\d+)?\s*(?:{_UNIT_PATTERN})?
        | { _UNIT_PATTERN }
        | q\d+h:?prn
        | q\d+h
        | qam:?prn
        | qpm:?prn
        | qhs:?prn
        | bid:?prn
        | tid:?prn
        | qid:?prn
        | [a-z]{{1,6}}\.[a-z.]+
        | [a-z][a-z0-9/-]*
        | \+
        | x\d+
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extend_medication_span_end(text: str, offset: int, max_chars: int = 120) -> int:
    max_end = min(len(text), offset + max_chars)
    tail = text[offset:max_end]
    stop = _first_medication_stop(tail)
    candidate = tail[:stop]
    match = _TAIL_TOKEN_RE.match(candidate)
    if not match:
        return offset

    accepted_end = 0
    for token_match in re.finditer(r"\S+", match.group(0)):
        token = token_match.group(0)
        if not _is_allowed_tail_token(normalize_prescription_text(token)):
            break
        accepted_end = token_match.end()
    if accepted_end <= 0:
        return offset
    return offset + accepted_end


def normalize_prescription_text(text: str) -> str:
    key = normalize_key(text)
    key = re.sub(r"\b(mg|mcg|g|ml|iu)\s*/\s*ml\b", r"\1/ml", key)
    key = re.sub(rf"(\d)(mg/ml|mcg/ml|mg|mcg|g|ml|iu|units?)\b", r"\1 \2", key)
    key = _Q_INTERVAL_RE.sub(r"q\1h:prn", key)
    key = _Q_INTERVAL_PLAIN_RE.sub(r"q\1h", key)
    for pattern, replacement in _ABBREVIATION_REPLACEMENTS:
        key = pattern.sub(replacement, key)
    key = re.sub(r"\b(qam|qpm|qhs|bid|tid|qid)\s*:?\s*prn\b", r"\1:prn", key)
    return " ".join(key.split())


def strip_drug_count(text: str) -> str:
    key = normalize_prescription_text(text)
    key = re.sub(r"\s+x\s*\d+(?:\s+(?:ngay|day|days))?\b.*$", "", key)
    return " ".join(key.split())


def strip_drug_route_frequency(text: str) -> str:
    tokens = normalize_prescription_text(text).split()
    kept: list[str] = []
    for token in tokens:
        base = token.split(":", 1)[0]
        if (
            base in ROUTE_TOKENS
            or base in FREQUENCY_TOKENS
            or token in COUNT_TOKENS
            or re.fullmatch(r"q\d+h:?prn?", token)
        ):
            break
        kept.append(token)
    return " ".join(kept)


def strip_drug_release_tokens(text: str) -> str:
    tokens = [
        token
        for token in normalize_prescription_text(text).split()
        if token not in {"xl", "xr", "er", "sr", "cr", "extended", "release"}
    ]
    return " ".join(tokens)


def strip_drug_modifiers(text: str) -> str:
    tokens = normalize_prescription_text(text).split()
    kept: list[str] = []
    for token in tokens:
        if (
            token in FORM_TOKENS
            or token in ROUTE_TOKENS
            or token in FREQUENCY_TOKENS
            or token in COUNT_TOKENS
            or any(ch.isdigit() for ch in token)
        ):
            break
        kept.append(token)
    return " ".join(kept)


def medication_match_score(query_text: str, candidate_text: str) -> float:
    query = normalize_prescription_text(query_text)
    candidate = normalize_prescription_text(candidate_text)
    score = 0.0

    query_strengths = _strengths(query)
    candidate_strengths = _strengths(candidate)
    if query_strengths:
        if query_strengths & candidate_strengths:
            score += 0.10
        elif candidate_strengths:
            score -= 0.12

    query_forms = _form_tokens(query)
    candidate_forms = _form_tokens(candidate)
    if query_forms and query_forms & candidate_forms:
        score += 0.03
    if "oral" in query.split() and ("oral" in candidate.split() or "tablet" in candidate_forms or "capsule" in candidate_forms):
        score += 0.02
    return score


def _first_medication_stop(text: str) -> int:
    stops = [len(text)]
    lower = text.casefold()
    for token in ("\n", ";", ",", " điều trị ", " dieu tri ", " for "):
        index = lower.find(token)
        if index >= 0:
            stops.append(index)
    return min(stops)


def _is_allowed_prescription_tail(text: str) -> bool:
    tokens = [normalize_prescription_text(token) for token in text.split()]
    if not tokens:
        return False
    return all(_is_allowed_tail_token(token) for token in tokens)


def _is_allowed_tail_token(token: str) -> bool:
    if not token:
        return False
    base = token.split(":", 1)[0]
    return bool(
        _TAIL_ALLOWED_RE.match(token)
        and (
            any(ch.isdigit() for ch in token)
            or token in UNIT_TOKENS
            or token in ROUTE_TOKENS
            or base in FREQUENCY_TOKENS
            or token in FORM_TOKENS
            or token in COUNT_TOKENS
            or token in {"+", "oral"}
        )
    )


def _strengths(text: str) -> set[str]:
    values: set[str] = set()
    for match in _STRENGTH_RE.finditer(normalize_prescription_text(text)):
        values.add(match.group(0).replace(",", ".").replace(" ", ""))
    return values


def _form_tokens(text: str) -> set[str]:
    return {token for token in normalize_prescription_text(text).split() if token in FORM_TOKENS}
