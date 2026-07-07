"""Candidate retrieval over the local ontology index."""

from __future__ import annotations

from functools import lru_cache

from medkg.config import CODED_TYPES
from medkg.ontology import OntologyIndex


class CandidateRetriever:
    def __init__(self, index: OntologyIndex):
        self.index = index

    @lru_cache(maxsize=8192)
    def candidates_for(self, text: str, concept_type: str, limit: int = 5) -> tuple[str, ...]:
        if concept_type not in CODED_TYPES:
            return ()
        entries = self.index.lookup(text, concept_type, limit=limit)
        return tuple(entry.code for entry in entries)

