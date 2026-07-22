"""Build a provenance-rich candidate alias lexicon from official training gold."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import CODED_TYPES, TYPE_DIAGNOSIS
from core.text import normalize_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input_part2" / "input" / "input")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "data" / "candidates")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "candidates" / "candidate_training_lexicon.json",
    )
    args = parser.parse_args(argv)

    payload = build_training_lexicon(
        args.gold_dir.resolve(),
        args.candidate_dir.resolve(),
        args.input_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "aliases": len(payload["aliases"]),
                "supplemental_records": len(payload["supplemental_records"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_training_lexicon(
    gold_dir: Path,
    candidate_dir: Path,
    input_dir: Path | None = None,
) -> dict[str, Any]:
    alias_codes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    alias_code_files: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    alias_empty: Counter[tuple[str, str]] = Counter()
    alias_files: dict[tuple[str, str], set[str]] = defaultdict(set)
    alias_sets: dict[tuple[str, str], Counter[tuple[str, ...]]] = defaultdict(Counter)
    profile_sets: dict[tuple[str, str, str], Counter[tuple[str, ...]]] = defaultdict(Counter)
    profile_files: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    code_names: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    gold_files = sorted(gold_dir.glob("*.json"), key=lambda path: _natural_key(path.name))
    for path in gold_files:
        source_text = ""
        if input_dir is not None:
            input_path = input_dir / f"{path.stem}.txt"
            if input_path.exists():
                source_text = input_path.read_text(encoding="utf-8")
        for item in _load_concepts(path):
            concept_type = str(item.get("type") or "")
            alias = normalize_key(str(item.get("text") or ""))
            if concept_type not in CODED_TYPES or not alias:
                continue
            key = (concept_type, alias)
            codes = tuple(
                dict.fromkeys(
                    str(code).strip()
                    for code in item.get("candidates") or ()
                    if str(code).strip()
                )
            )
            alias_files[key].add(path.name)
            alias_sets[key][tuple(sorted(codes))] += 1
            profile = _line_profile(source_text, item.get("position"))
            profile_sets[(concept_type, alias, profile)][tuple(sorted(codes))] += 1
            profile_files[(concept_type, alias, profile)].add(path.name)
            if not codes:
                alias_empty[key] += 1
                continue
            mention = str(item.get("text") or "").strip()
            for code in codes:
                alias_codes[key][code] += 1
                alias_code_files[(concept_type, alias, code)].add(path.name)
                code_names[(concept_type, code)][mention] += 1

    aliases: list[dict[str, Any]] = []
    for (concept_type, alias), code_counts in sorted(alias_codes.items()):
        ranked_codes = sorted(
            code_counts,
            key=lambda code: (
                -len(alias_code_files[(concept_type, alias, code)]),
                -code_counts[code],
                code,
            ),
        )
        aliases.append(
            {
                "type": concept_type,
                "alias_norm": alias,
                "candidates": [
                    {
                        "code": code,
                        "support": code_counts[code],
                        "file_support": len(alias_code_files[(concept_type, alias, code)]),
                        "files": sorted(
                            alias_code_files[(concept_type, alias, code)],
                            key=_natural_key,
                        ),
                    }
                    for code in ranked_codes
                ],
                "empty_support": alias_empty[(concept_type, alias)],
                "file_support": len(alias_files[(concept_type, alias)]),
                "candidate_set_distribution": [
                    {"candidates": list(codes), "support": support}
                    for codes, support in sorted(
                        alias_sets[(concept_type, alias)].items(),
                        key=lambda row: (-row[1], row[0]),
                    )
                ],
                "profiles": [
                    {
                        "name": profile,
                        "support": sum(profile_sets[(concept_type, alias, profile)].values()),
                        "file_support": len(profile_files[(concept_type, alias, profile)]),
                        "files": sorted(
                            profile_files[(concept_type, alias, profile)],
                            key=_natural_key,
                        ),
                        "preferred_candidates": (
                            None if preferred is None else list(preferred)
                        ),
                        "candidate_set_distribution": [
                            {"candidates": list(codes), "support": support}
                            for codes, support in sorted(
                                profile_sets[(concept_type, alias, profile)].items(),
                                key=lambda row: (-row[1], row[0]),
                            )
                        ],
                    }
                    for profile in ("short_line", "long_line")
                    if profile_sets[(concept_type, alias, profile)]
                    for preferred in [
                        _preferred_candidate_set(
                            profile_sets[(concept_type, alias, profile)]
                        )
                    ]
                ],
            }
        )

    existing_codes = _existing_codes(candidate_dir)
    supplemental_records: list[dict[str, Any]] = []
    for (concept_type, code), names in sorted(code_names.items()):
        if (concept_type, code) in existing_codes:
            continue
        name = names.most_common(1)[0][0]
        supplemental_records.append(
            {
                "type": concept_type,
                "code": code,
                "name": name,
                "system": "ICD10" if concept_type == TYPE_DIAGNOSIS else "RxNorm",
                "priority": 20,
                "archive": False,
                "ttys": [],
                "provenance": "input_part2 official training ground truth",
            }
        )

    return {
        "version": 1,
        "purpose": "Training-derived retrieval aliases; occurrence emission remains a separate decision.",
        "provenance": {
            "gold_dir": str(gold_dir.relative_to(ROOT) if gold_dir.is_relative_to(ROOT) else gold_dir),
            "files": len(gold_files),
            "policy": "All positive aliases are retained with code/file support and competing empty labels.",
        },
        "summary": {
            "aliases": len(aliases),
            "supplemental_records": len(supplemental_records),
        },
        "supplemental_records": supplemental_records,
        "aliases": aliases,
    }


def _load_concepts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        concepts = payload.get("concepts")
        if isinstance(concepts, list):
            return [item for item in concepts if isinstance(item, dict)]
    return []


def _existing_codes(candidate_dir: Path) -> set[tuple[str, str]]:
    output: set[tuple[str, str]] = set()
    for file_name in ("icd10_candidates.jsonl", "rxnorm_candidates.jsonl"):
        path = candidate_dir / file_name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                concept_type = str(row.get("type") or "")
                code = str(row.get("code") or "").strip()
                if concept_type in CODED_TYPES and code:
                    output.add((concept_type, code))
    return output


def _natural_key(value: str) -> tuple[object, ...]:
    import re

    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _line_profile(source_text: str, position: object) -> str:
    if (
        not source_text
        or not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(value, int) for value in position)
    ):
        return "short_line"
    start, end = position
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", end)
    if line_end < 0:
        line_end = len(source_text)
    return "short_line" if line_end - line_start < 300 else "long_line"


def _preferred_candidate_set(
    distribution: Counter[tuple[str, ...]],
) -> tuple[str, ...] | None:
    if not distribution:
        return None
    options = set(distribution)
    options.add(())

    def utility(predicted: tuple[str, ...]) -> float:
        predicted_set = set(predicted)
        total = 0.0
        for expected, support in distribution.items():
            expected_set = set(expected)
            if not expected_set and not predicted_set:
                score = 1.0
            elif expected_set or predicted_set:
                score = len(expected_set & predicted_set) / len(expected_set | predicted_set)
            else:
                score = 1.0
            total += support * (len(expected_set) + 1) * score
        return total

    utilities = {option: utility(option) for option in options}
    best_utility = max(utilities.values())
    best_options = [
        option for option, option_utility in utilities.items()
        if option_utility == best_utility
    ]
    if len(best_options) != 1:
        return None
    return best_options[0]


if __name__ == "__main__":
    raise SystemExit(main())
