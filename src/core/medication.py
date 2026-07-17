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
    "powder",
    "drop",
    "drops",
    "drip",
    "extended",
    "release",
    "delayed",
    "enteric",
    "coated",
    "chewable",
    "effervescent",
    "disintegrating",
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
_MEDICATION_SPELLING_REPLACEMENTS = (
    (re.compile(r"\basa\b"), "aspirin"),
    (re.compile(r"\bciprofloxaxin\b"), "ciprofloxacin"),
    (re.compile(r"\baldactol\b"), "spironolactone"),
    (re.compile(r"\bnifedipin\b"), "nifedipine"),
    (re.compile(r"\bspironolacton\b"), "spironolactone"),
    (re.compile(r"\bomeprazol\b"), "omeprazole"),
    (re.compile(r"\brisperidon\b"), "risperidone"),
    # The Vietnamese market spelling is Forxiga; RxNorm stores Farxiga.
    (re.compile(r"\bforxiga\b"), "farxiga"),
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
    for pattern, replacement in _MEDICATION_SPELLING_REPLACEMENTS:
        key = pattern.sub(replacement, key)
    key = re.sub(r"\b(qam|qpm|qhs|bid|tid|qid)\s*:?\s*prn\b", r"\1:prn", key)
    return " ".join(key.split())


def medication_lookup_keys(text: str) -> tuple[str, ...]:
    """Return ingredient and brand keys without losing parenthetical drug names."""

    segments = [text, _strip_leading_dose_route(text)]
    leading = re.split(r"[\(\[]", text, maxsplit=1)[0]
    if leading != text:
        segments.append(leading)
        segments.append(_strip_leading_dose_route(leading))
    segments.extend(re.findall(r"[\(\[]([^\)\]]+)[\)\]]", text))

    keys: list[str] = []
    for segment in segments:
        key = medication_ingredient_key(segment)
        key = re.sub(r"^(?:duoi dang|base|as)\s+", "", key)
        key = re.sub(r"^(?:thuoc\s+hit|thuoc)\s+", "", key)
        if len(key) >= 3 and key not in keys:
            keys.append(key)
    return tuple(keys)


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
            score += 0.25
        elif candidate_strengths:
            score -= 0.60
        else:
            score -= 0.30

    query_forms = _form_tokens(query)
    candidate_forms = _form_tokens(candidate)
    if query_forms and query_forms & candidate_forms:
        score += 0.03
    release_tokens = {"xl", "xr", "er", "sr", "cr", "extended", "release", "delayed", "enteric", "coated"}
    query_release = query_forms & release_tokens
    candidate_release = candidate_forms & release_tokens
    if candidate_release and not query_release:
        score -= 0.18
    elif query_release and not candidate_release:
        score -= 0.20
    specialized_forms = {"chewable", "effervescent", "disintegrating"}
    if candidate_forms & specialized_forms and not query_forms & specialized_forms:
        score -= 0.12
    query_tokens = set(query.split())
    if "/" in candidate and "/" not in query:
        score -= 0.50
    if query_tokens & {"oral", "po"} and (
        "oral" in candidate.split() or "tablet" in candidate_forms or "capsule" in candidate_forms
    ):
        score += 0.08
    brand_match = re.search(r"\[([^\]]+)\]", candidate_text)
    if query_strengths and not query_forms:
        if "tablet" in candidate_forms:
            score += 0.02 if brand_match else 0.10
        elif "capsule" in candidate_forms:
            score += 0.02 if brand_match else 0.08
        elif candidate_forms & {"powder", "solution", "suspension", "injection"}:
            score -= 0.05
    if brand_match:
        brand_key = normalize_key(brand_match.group(1))
        if brand_key and brand_key not in query:
            score -= 0.05
    return score


def medication_ingredient_key(text: str) -> str:
    """Return the ingredient/brand prefix used for local product retrieval."""

    key = strip_drug_modifiers(text)
    key = re.sub(r"\s*\[[^\]]+\]\s*$", "", key)
    return " ".join(key.split())


def medication_has_strength(text: str) -> bool:
    return bool(_strengths(text))


def medication_strength_relation(query_text: str, candidate_text: str) -> str:
    """Classify strength compatibility for product-level RxNorm mapping."""

    query_strengths = _strengths(query_text)
    candidate_strengths = _strengths(candidate_text)
    if not query_strengths:
        return "not_requested"
    if query_strengths & candidate_strengths:
        return "match"
    if candidate_strengths:
        return "conflict"
    return "missing"


def medication_tty_score(query_text: str, ttys: tuple[str, ...]) -> float:
    tty_set = {tty.upper() for tty in ttys}
    has_strength = medication_has_strength(query_text)
    query_tokens = set(normalize_prescription_text(query_text).split())
    has_form_or_route = bool(query_tokens & (FORM_TOKENS | ROUTE_TOKENS))
    if has_strength:
        if not has_form_or_route:
            if "SBDC" in tty_set:
                return 0.33
            if "SCDC" in tty_set:
                return 0.31
            if "SCD" in tty_set:
                return 0.30
            if "PSN" in tty_set:
                return 0.20
            if "SBD" in tty_set:
                return 0.18
        if "SCD" in tty_set:
            return 0.27
        if "PSN" in tty_set:
            return 0.26
        if "SBDC" in tty_set:
            return 0.24
        if "SBD" in tty_set:
            return 0.21
        if "SCDC" in tty_set:
            return 0.20
        if "SY" in tty_set:
            return 0.18
        if "SCDC" in tty_set:
            return 0.05
        if tty_set & {"IN", "PIN", "MIN", "BN"}:
            return -0.35
    else:
        if tty_set & {"IN", "PIN", "MIN", "BN"}:
            return 0.15
        if tty_set & {"SCD", "SBD", "SCDC"}:
            return -0.08
    return 0.0


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
        raw = match.group(0).replace(",", ".")
        values.add(_canonical_strength(raw))
    return values


def _strip_leading_dose_route(text: str) -> str:
    key = normalize_prescription_text(text)
    tokens = key.split()
    while tokens:
        token = tokens[0]
        base = token.split(":", 1)[0]
        if (
            any(ch.isdigit() for ch in token)
            or token in UNIT_TOKENS
            or token in ROUTE_TOKENS
            or base in FREQUENCY_TOKENS
            or token in COUNT_TOKENS
        ):
            tokens.pop(0)
            continue
        break
    return " ".join(tokens)


def _canonical_strength(value: str) -> str:
    compact = value.replace(" ", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(mg|g|mcg|ml|iu|unit|units|%)", compact)
    if not match:
        return compact
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "g":
        amount *= 1000
        unit = "mg"
    if amount.is_integer():
        amount_text = str(int(amount))
    else:
        amount_text = ("%f" % amount).rstrip("0").rstrip(".")
    return f"{amount_text}{unit}"


def _form_tokens(text: str) -> set[str]:
    return {token for token in normalize_prescription_text(text).split() if token in FORM_TOKENS}
