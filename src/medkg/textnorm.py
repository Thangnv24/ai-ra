"""Compatibility wrapper for text normalization helpers."""

from medkg.normalization import compact_key, normalize_key, similarity, strip_accents, token_count, trim_span_text

__all__ = [
    "compact_key",
    "normalize_key",
    "similarity",
    "strip_accents",
    "token_count",
    "trim_span_text",
]
