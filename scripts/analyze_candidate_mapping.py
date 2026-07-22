"""Analyze ICD-10/RxNorm mapping with gold mention spans and types."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.config import CODED_TYPES, TYPE_DIAGNOSIS, TYPE_DRUG
from knowledge.candidate_policy import load_candidate_emission_policy
from knowledge.candidates import CandidateHit, load_slim_candidate_index
from knowledge.retrieval import (
    _candidate_eligible,
    _select_diagnosis_codes,
    _select_drug_code,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate mapping independently with gold spans and types."
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input_part2" / "input" / "input")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "data" / "candidates")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=ROOT / "data" / "candidates" / "candidate_emission_policy.json",
    )
    parser.add_argument("--retrieval-limit", type=int, default=30)
    parser.add_argument("--min-training-file-support", type=int, default=1)
    parser.add_argument("--top-errors", type=int, default=20)
    parser.add_argument("--include-items", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    report = analyze_candidate_mapping(
        input_dir=args.input_dir,
        gold_dir=args.gold_dir,
        candidate_dir=args.candidate_dir,
        policy_path=args.policy_path,
        retrieval_limit=max(5, args.retrieval_limit),
        min_training_file_support=max(1, args.min_training_file_support),
        top_errors=max(0, args.top_errors),
        include_items=args.include_items,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(payload.encode("utf-8") + b"\n")
    else:
        print(payload)
    return 0


def analyze_candidate_mapping(
    *,
    input_dir: Path,
    gold_dir: Path,
    candidate_dir: Path,
    policy_path: Path | None = None,
    retrieval_limit: int = 30,
    min_training_file_support: int = 1,
    top_errors: int = 20,
    include_items: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    load_started = time.perf_counter()
    slim_index = load_slim_candidate_index(
        candidate_dir.resolve(),
        min_training_file_support=max(1, min_training_file_support),
    )
    policy = load_candidate_emission_policy(policy_path.resolve()) if policy_path else None
    index_load_seconds = time.perf_counter() - load_started
    stats = {"all": _new_stats(), TYPE_DIAGNOSIS: _new_stats(), TYPE_DRUG: _new_stats()}
    profile_stats = {
        f"{concept_type}:{line_profile}": _new_stats()
        for concept_type in (TYPE_DIAGNOSIS, TYPE_DRUG)
        for line_profile in ("short_line", "long_line")
    }
    errors: dict[str, Counter[tuple[str, str, tuple[str, ...], tuple[str, ...]]]] = {
        TYPE_DIAGNOSIS: Counter(),
        TYPE_DRUG: Counter(),
    }
    file_scores: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    lookup_cache: dict[tuple[str, str], list[CandidateHit]] = {}
    validation = Counter()
    item_rows: list[dict[str, Any]] = []

    gold_files = sorted(gold_dir.resolve().glob("*.json"), key=_natural_key)
    for gold_path in gold_files:
        input_path = input_dir.resolve() / f"{gold_path.stem}.txt"
        if not input_path.exists():
            validation["missing_input_files"] += 1
            continue
        source_text = input_path.read_text(encoding="utf-8")
        concepts = _load_concepts(gold_path)
        for item_index, item in enumerate(concepts):
            concept_type = str(item.get("type") or "")
            if concept_type not in CODED_TYPES:
                continue
            position = item.get("position")
            if not _valid_position(position) or source_text[position[0] : position[1]] != item.get("text"):
                validation["invalid_gold_offsets"] += 1
                continue

            start, end = position
            mention = str(item.get("text") or "")
            expected = tuple(str(code) for code in item.get("candidates") or () if code)
            cache_key = (mention, concept_type)
            hits = lookup_cache.get(cache_key)
            if hits is None:
                hits = slim_index.lookup(mention, concept_type, limit=retrieval_limit)
                lookup_cache[cache_key] = hits

            eligible = _candidate_eligible(concept_type, source_text, start, end)
            ungated = _select_from_hits(mention, concept_type, hits)
            predicted = ungated if eligible else (
                slim_index.candidates_for_profile(
                    mention,
                    concept_type,
                    source_text,
                    start,
                    end,
                )
                or ()
            )
            if policy is not None:
                predicted = policy.apply(
                    mention,
                    concept_type,
                    predicted,
                    source_text=source_text,
                    start=start,
                    end=end,
                    profile_candidates=slim_index.candidates_for_profile(
                        mention,
                        concept_type,
                        source_text,
                        start,
                        end,
                    ),
                    assertions=tuple(str(value) for value in item.get("assertions") or () if value),
                )
            outcome = _classify_outcome(
                expected,
                predicted,
                hits,
                eligible=eligible,
                records=slim_index.records,
                concept_type=concept_type,
            )
            line_profile = _line_profile(source_text, start, end)
            for bucket in (
                stats["all"],
                stats[concept_type],
                profile_stats[f"{concept_type}:{line_profile}"],
            ):
                _add_result(
                    bucket,
                    expected=expected,
                    predicted=predicted,
                    hits=hits,
                    outcome=outcome,
                    records=slim_index.records,
                    concept_type=concept_type,
                )

            score = _jaccard(set(expected), set(predicted))
            weight = len(set(expected)) + 1
            file_scores[gold_path.name][0] += score * weight
            file_scores[gold_path.name][1] += weight
            if outcome not in {"exact_positive", "correct_abstain"}:
                errors[concept_type][(outcome, mention, expected, predicted)] += 1
            if include_items:
                item_rows.append(
                    {
                        "file": gold_path.name,
                        "position": [start, end],
                        "text": mention,
                        "type": concept_type,
                        "line_profile": line_profile,
                        "assertions": list(item.get("assertions") or ()),
                        "expected": list(expected),
                        "predicted": list(predicted),
                        "ungated": list(ungated),
                        "eligible": eligible,
                        "outcome": outcome,
                    }
                )

    report = {
        "files": len(gold_files),
        "input_dir": str(input_dir.resolve()),
        "gold_dir": str(gold_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "policy_path": str(policy_path.resolve()) if policy_path else None,
        "retrieval_limit": retrieval_limit,
        "min_training_file_support": min_training_file_support,
        "validation": dict(validation),
        "index": {
            "records": len(slim_index.records),
            "aliases": len(slim_index.aliases),
            "load_seconds": round(index_load_seconds, 3),
            "unique_queries": len(lookup_cache),
        },
        "summary": {
            key: _summarize(value)
            for key, value in stats.items()
        },
        "line_profiles": {
            key: _summarize(value)
            for key, value in profile_stats.items()
        },
        "top_errors": {
            concept_type: [
                {
                    "count": count,
                    "outcome": key[0],
                    "text": key[1],
                    "expected": list(key[2]),
                    "predicted": list(key[3]),
                }
                for key, count in errors[concept_type].most_common(top_errors)
            ]
            for concept_type in (TYPE_DIAGNOSIS, TYPE_DRUG)
        },
        "worst_files": [
            {"file": file_name, "weighted_jaccard": round(numerator / denominator, 6)}
            for file_name, (numerator, denominator) in sorted(
                file_scores.items(),
                key=lambda item: item[1][0] / item[1][1] if item[1][1] else 1.0,
            )[:20]
            if denominator
        ],
        "timing_seconds": round(time.perf_counter() - started, 3),
    }
    if include_items:
        report["items"] = item_rows
    return report


def _new_stats() -> dict[str, Any]:
    return {
        "mentions": 0,
        "gold_nonempty": 0,
        "predicted_nonempty": 0,
        "exact": 0,
        "weighted_numerator": 0.0,
        "weighted_denominator": 0.0,
        "all_empty_numerator": 0.0,
        "positive_exact": 0,
        "positive_overlap": 0,
        "negative_abstain": 0,
        "kb_all_codes": 0,
        "hit_at_1": 0,
        "hit_at_5": 0,
        "hit_at_30": 0,
        "outcomes": Counter(),
        "predicted_sources": Counter(),
    }


def _add_result(
    stats: dict[str, Any],
    *,
    expected: tuple[str, ...],
    predicted: tuple[str, ...],
    hits: list[CandidateHit],
    outcome: str,
    records: dict[tuple[str, str], Any],
    concept_type: str,
) -> None:
    expected_set = set(expected)
    predicted_set = set(predicted)
    hit_codes = [hit.record.code for hit in hits]
    score = _jaccard(expected_set, predicted_set)
    weight = len(expected_set) + 1

    stats["mentions"] += 1
    stats["gold_nonempty"] += bool(expected_set)
    stats["predicted_nonempty"] += bool(predicted_set)
    stats["exact"] += expected_set == predicted_set
    stats["weighted_numerator"] += score * weight
    stats["weighted_denominator"] += weight
    stats["all_empty_numerator"] += weight if not expected_set else 0
    stats["outcomes"][outcome] += 1

    if not expected_set:
        stats["negative_abstain"] += not predicted_set
        if predicted_set:
            source_by_code = {hit.record.code: hit.source for hit in hits}
            for code in predicted_set:
                stats["predicted_sources"][source_by_code.get(code, "not_in_retrieval")] += 1
        return

    stats["positive_exact"] += expected_set == predicted_set
    stats["positive_overlap"] += bool(expected_set & predicted_set)
    stats["kb_all_codes"] += all((concept_type, code) in records for code in expected_set)
    stats["hit_at_1"] += bool(expected_set & set(hit_codes[:1]))
    stats["hit_at_5"] += bool(expected_set & set(hit_codes[:5]))
    stats["hit_at_30"] += bool(expected_set & set(hit_codes[:30]))


def _classify_outcome(
    expected: tuple[str, ...],
    predicted: tuple[str, ...],
    hits: list[CandidateHit],
    *,
    eligible: bool,
    records: dict[tuple[str, str], Any],
    concept_type: str,
) -> str:
    expected_set = set(expected)
    predicted_set = set(predicted)
    if not expected_set:
        return "correct_abstain" if not predicted_set else "false_positive"
    if expected_set == predicted_set:
        return "exact_positive"
    if expected_set & predicted_set:
        return "partial_positive"
    if not eligible:
        return "context_gate_abstain"
    available = {code for code in expected_set if (concept_type, code) in records}
    if not available:
        return "kb_missing"
    hit_codes = {hit.record.code for hit in hits}
    if not (available & hit_codes):
        return "retrieval_miss" if not predicted_set else "retrieval_wrong"
    return "selector_abstain" if not predicted_set else "selector_wrong"


def _select_from_hits(
    mention: str,
    concept_type: str,
    hits: list[CandidateHit],
) -> tuple[str, ...]:
    if concept_type == TYPE_DRUG:
        return _select_drug_code(mention, hits)
    return _select_diagnosis_codes(mention, hits, limit=5)


def _line_profile(source_text: str, start: int, end: int) -> str:
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", end)
    if line_end < 0:
        line_end = len(source_text)
    return "short_line" if line_end - line_start < 300 else "long_line"


def _summarize(stats: dict[str, Any]) -> dict[str, Any]:
    mentions = stats["mentions"]
    positives = stats["gold_nonempty"]
    negatives = mentions - positives
    weighted_score = _ratio(stats["weighted_numerator"], stats["weighted_denominator"])
    all_empty_score = _ratio(stats["all_empty_numerator"], stats["weighted_denominator"])
    return {
        "mentions": mentions,
        "gold_nonempty": positives,
        "gold_nonempty_rate": round(_ratio(positives, mentions), 6),
        "predicted_nonempty": stats["predicted_nonempty"],
        "predicted_nonempty_rate": round(_ratio(stats["predicted_nonempty"], mentions), 6),
        "exact_set_rate": round(_ratio(stats["exact"], mentions), 6),
        "weighted_jaccard": round(weighted_score, 6),
        "all_empty_weighted_jaccard": round(all_empty_score, 6),
        "lift_over_all_empty": round(weighted_score - all_empty_score, 6),
        "positive_exact_rate": round(_ratio(stats["positive_exact"], positives), 6),
        "positive_overlap_rate": round(_ratio(stats["positive_overlap"], positives), 6),
        "negative_abstain_rate": round(_ratio(stats["negative_abstain"], negatives), 6),
        "kb_all_codes_coverage": round(_ratio(stats["kb_all_codes"], positives), 6),
        "retrieval_hit_at_1": round(_ratio(stats["hit_at_1"], positives), 6),
        "retrieval_hit_at_5": round(_ratio(stats["hit_at_5"], positives), 6),
        "retrieval_hit_at_30": round(_ratio(stats["hit_at_30"], positives), 6),
        "outcomes": dict(stats["outcomes"]),
        "false_positive_sources": dict(stats["predicted_sources"].most_common()),
    }


def _load_concepts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def _valid_position(position: Any) -> bool:
    return bool(
        isinstance(position, list)
        and len(position) == 2
        and all(isinstance(value, int) for value in position)
        and 0 <= position[0] <= position[1]
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (sys.maxsize, path.name)


if __name__ == "__main__":
    raise SystemExit(main())
