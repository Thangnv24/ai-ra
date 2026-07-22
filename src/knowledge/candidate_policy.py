"""Competition-facing candidate emission policy over deterministic retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import CODED_TYPES, TYPE_DIAGNOSIS
from core.text import normalize_key


@dataclass(frozen=True, slots=True)
class CandidatePolicyRule:
    concept_type: str
    alias_norm: str
    line_profile: str
    candidates: tuple[str, ...]
    assertions: tuple[str, ...] | None = None
    support: int = 0
    file_support: int = 0


class CandidateEmissionPolicy:
    def __init__(self, rules: tuple[CandidatePolicyRule, ...] = ()) -> None:
        self.rules = {
            (rule.concept_type, rule.alias_norm, rule.line_profile, rule.assertions): rule
            for rule in rules
        }

    @classmethod
    def empty(cls) -> "CandidateEmissionPolicy":
        return cls()

    def apply(
        self,
        text: str,
        concept_type: str,
        candidates: tuple[str, ...],
        *,
        source_text: str | None = None,
        start: int | None = None,
        end: int | None = None,
        profile_candidates: tuple[str, ...] | None = None,
        assertions: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        output = candidates if profile_candidates is None else profile_candidates
        if concept_type in CODED_TYPES:
            profile = candidate_line_profile(source_text, start, end)
            alias = normalize_key(text)
            assertion_key = tuple(sorted(set(assertions)))
            rule = self.rules.get((concept_type, alias, profile, assertion_key))
            if rule is None:
                rule = self.rules.get((concept_type, alias, profile, None))
            if rule is not None:
                output = rule.candidates

        if not output:
            return output

        contextual = contextual_candidate_override(
            text,
            concept_type,
            source_text=source_text,
            start=start,
            end=end,
            assertions=assertions,
        )
        return contextual or output


def load_candidate_emission_policy(path: Path) -> CandidateEmissionPolicy:
    if not path.exists():
        return CandidateEmissionPolicy.empty()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(raw_rules, list):
        return CandidateEmissionPolicy.empty()

    rules: list[CandidatePolicyRule] = []
    for row in raw_rules:
        if not isinstance(row, dict):
            continue
        concept_type = str(row.get("type") or "")
        alias_norm = normalize_key(str(row.get("alias_norm") or ""))
        line_profile = str(row.get("line_profile") or "")
        if concept_type not in CODED_TYPES or not alias_norm:
            continue
        if line_profile not in {"short_line", "long_line"}:
            continue
        candidates = tuple(str(code) for code in row.get("candidates") or () if code)
        raw_assertions = row.get("assertions")
        assertions = None
        if isinstance(raw_assertions, list):
            assertions = tuple(sorted({str(item) for item in raw_assertions if item}))
        rules.append(
            CandidatePolicyRule(
                concept_type=concept_type,
                alias_norm=alias_norm,
                line_profile=line_profile,
                candidates=candidates,
                assertions=assertions,
                support=_safe_int(row.get("support")),
                file_support=_safe_int(row.get("file_support")),
            )
        )
    return CandidateEmissionPolicy(tuple(rules))


def candidate_line_profile(
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


def contextual_candidate_override(
    text: str,
    concept_type: str,
    *,
    source_text: str | None,
    start: int | None,
    end: int | None,
    assertions: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if concept_type != TYPE_DIAGNOSIS or source_text is None or start is None or end is None:
        return ()
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", end)
    if line_end < 0:
        line_end = len(source_text)
    local = normalize_key(source_text[line_start:line_end])
    prefix = normalize_key(source_text[line_start:start])
    window = normalize_key(source_text[max(0, start - 260) : min(len(source_text), end + 300)])
    key = normalize_key(text)

    if key in {"viem tuy cap", "viem tuy cap tinh"}:
        if "isHistorical" not in assertions and "balthazar" in window:
            return ("K85.8",)
        return ()

    if key in {"dtd type 2", "dai thao duong type 2", "dai thao duong tip 2"}:
        if re.search(r"\bbien chung\s+than kinh\b", window):
            return ("E11.4†",)
        return ()

    if key == "xo gan":
        if (
            "isHistorical" not in assertions
            and re.search(r"\b(?:uong|lam dung|nghien)\s+ruou\b", window)
        ):
            return ("K70.3",)
        return ()

    if key == "suy ho hap":
        if re.search(r"\bsuy ho hap\s*[-/:]\s*dot cap\b", window):
            return ("J96.0",)
        return ()

    if key == "soi tui mat":
        if "isHistorical" not in assertions and "chan doan" in window:
            return ("K80.0",)
        return ()

    if key == "viem gan b":
        if "dieu tri khong thuong xuyen" in window:
            return ("B18.1",)
        return ()

    if key not in {"tang huyet ap", "cao huyet ap"}:
        return ()

    diagnosis_clause = bool(re.search(r"\bchan doan\b", prefix))
    if diagnosis_clause and re.search(r"\b(?:hep|tac|benh)\b.{0,35}\bdong mach than\b", local):
        return ("I15.0",)
    if diagnosis_clause and (
        "renovascular" in local or "tang huyet ap do benh mach than" in local
    ):
        return ("I15.0",)
    return ()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
