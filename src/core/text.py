"""Text normalization helpers for matching noisy clinical text."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w.%/+:-]+", re.UNICODE)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_marks.replace("\u0111", "d").replace("\u0110", "D")


def normalize_key(text: str) -> str:
    text = strip_accents(text).casefold()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_key(text))


def token_count(text: str) -> int:
    return len([tok for tok in normalize_key(text).split(" ") if tok])


def similarity(a: str, b: str) -> float:
    ak = normalize_key(a)
    bk = normalize_key(b)
    if not ak or not bk:
        return 0.0
    if ak == bk:
        return 1.0
    return SequenceMatcher(None, ak, bk).ratio()


def trim_span_text(text: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    return start, end, text[start:end]
