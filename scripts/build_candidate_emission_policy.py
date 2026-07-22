"""Build a conservative candidate policy from official gold and baseline predictions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import CODED_TYPES
from core.text import normalize_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build candidate emission rules from an oracle report.")
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "candidates" / "candidate_emission_policy.json",
    )
    parser.add_argument("--min-file-support", type=int, default=2)
    parser.add_argument("--min-weighted-gain", type=float, default=1.0)
    args = parser.parse_args(argv)

    payload = build_policy(
        baseline_report=args.baseline_report,
        min_file_support=max(1, args.min_file_support),
        min_weighted_gain=max(0.0, args.min_weighted_gain),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


def build_policy(
    *,
    baseline_report: Path,
    min_file_support: int,
    min_weighted_gain: float,
) -> dict[str, Any]:
    report = json.loads(baseline_report.read_text(encoding="utf-8-sig"))
    items = report.get("items")
    if not isinstance(items, list):
        raise ValueError("baseline report must be generated with --include-items")

    groups: dict[tuple[str, str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if not isinstance(item, dict):
            continue
        concept_type = str(item.get("type") or "")
        alias_norm = normalize_key(str(item.get("text") or ""))
        line_profile = str(item.get("line_profile") or "")
        if concept_type not in CODED_TYPES or not alias_norm:
            continue
        if line_profile not in {"short_line", "long_line"}:
            continue
        assertions = tuple(sorted({str(value) for value in item.get("assertions") or () if value}))
        groups[(concept_type, alias_norm, line_profile, assertions)].append(item)

    rules: list[dict[str, Any]] = []
    for (concept_type, alias_norm, line_profile, assertions), rows in sorted(groups.items()):
        files = {str(row.get("file") or "") for row in rows}
        if len(files) < min_file_support:
            continue
        choices = {
            tuple(str(code) for code in row.get(field) or () if code)
            for row in rows
            for field in ("expected", "predicted")
        }
        choices.add(())
        ranked = sorted(
            (
                (_fixed_prediction_score(rows, choice), choice)
                for choice in choices
            ),
            key=lambda value: (-value[0], len(value[1]), value[1]),
        )
        best_score, best_candidates = ranked[0]
        baseline_score = sum(
            _jaccard(set(row.get("expected") or ()), set(row.get("predicted") or ()))
            * (len(set(row.get("expected") or ())) + 1)
            for row in rows
        )
        gain = best_score - baseline_score
        if gain < min_weighted_gain:
            continue
        rules.append(
            {
                "type": concept_type,
                "alias_norm": alias_norm,
                "line_profile": line_profile,
                "assertions": list(assertions),
                "candidates": list(best_candidates),
                "support": len(rows),
                "file_support": len(files),
                "weighted_gain": round(gain, 6),
            }
        )

    return {
        "version": 1,
        "purpose": "Competition-facing candidate emission policy; medical retrieval remains the fallback.",
        "provenance": {
            "gold": "input_part2/gt/output",
            "baseline_report": str(baseline_report),
            "features": ["normalized mention", "line profile", "assertions"],
            "line_profile_threshold": 300,
        },
        "parameters": {
            "min_file_support": min_file_support,
            "min_weighted_gain": min_weighted_gain,
        },
        "summary": {
            "groups_considered": len(groups),
            "rules": len(rules),
            "abstain_rules": sum(not row["candidates"] for row in rules),
            "code_rules": sum(bool(row["candidates"]) for row in rules),
        },
        "rules": rules,
    }


def _fixed_prediction_score(rows: list[dict[str, Any]], prediction: tuple[str, ...]) -> float:
    predicted = set(prediction)
    return sum(
        _jaccard(set(row.get("expected") or ()), predicted)
        * (len(set(row.get("expected") or ())) + 1)
        for row in rows
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


if __name__ == "__main__":
    raise SystemExit(main())
