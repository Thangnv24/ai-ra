"""Evaluate annotation memory on held-out document-structure groups."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.config import CODED_TYPES
from extraction.annotation_memory import AnnotationMemory, MemoryEntry
from extraction.ner import MedicalNER
from extraction.sectioning import split_chunks
from extraction.span_verifier import SpanTypeVerifier
from scripts.build_annotation_memory import build_memory, natural_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leave one document-structure group out.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input_part2" / "input" / "input")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "external" / "annotation_memory_cv_report.json",
    )
    args = parser.parse_args(argv)
    report = evaluate(args.input_dir.resolve(), args.gold_dir.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def evaluate(input_dir: Path, gold_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    documents: dict[str, tuple[str, list[dict[str, Any]], str]] = {}
    groups: dict[str, set[str]] = {}
    for gold_path in sorted(gold_dir.glob("*.json"), key=natural_key):
        input_path = input_dir / f"{gold_path.stem}.txt"
        if not input_path.exists():
            continue
        text = input_path.read_text(encoding="utf-8-sig")
        raw = json.loads(gold_path.read_text(encoding="utf-8-sig"))
        gold = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        group = structure_group(text)
        documents[gold_path.stem] = (text, gold, group)
        groups.setdefault(group, set()).add(gold_path.stem)

    all_stems = set(documents)
    folds: list[dict[str, Any]] = []
    aggregate_baseline = SpanCounts()
    aggregate_memory = SpanCounts()
    aggregate_candidates = CandidateDecisionScores()
    for group, holdout_stems in sorted(groups.items()):
        train_stems = all_stems - holdout_stems
        rows, memory_stats = build_memory(
            [(input_dir, gold_dir)],
            include_stems=train_stems,
        )
        entries = [entry for row in rows if (entry := MemoryEntry.from_dict(row)) is not None]
        memory = AnnotationMemory(entries)
        ner = MedicalNER()
        baseline_verifier = SpanTypeVerifier()
        memory_verifier = SpanTypeVerifier(memory)
        baseline_counts = SpanCounts()
        memory_counts = SpanCounts()
        candidate_scores = CandidateDecisionScores()
        for stem in sorted(holdout_stems, key=lambda value: (int(value) if value.isdigit() else sys.maxsize, value)):
            text, gold, _ = documents[stem]
            gold_counter = Counter(
                (tuple(item.get("position") or ()), str(item.get("type") or ""))
                for item in gold
                if valid_position(item.get("position"))
            )
            rule_proposals = ner.propose(text)
            baseline_spans, _ = baseline_verifier.select(text, rule_proposals)
            memory_spans, _ = memory_verifier.select(text, [*rule_proposals, *memory.propose(text)])
            baseline_counts.add(gold_counter, span_counter(baseline_spans))
            memory_counts.add(gold_counter, span_counter(memory_spans))
            for item in gold:
                concept_type = str(item.get("type") or "")
                if concept_type not in CODED_TYPES:
                    continue
                decision = memory.candidate_decision(str(item.get("text") or ""), concept_type)
                candidate_scores.add(item.get("candidates") or (), decision)
        aggregate_baseline.merge(baseline_counts)
        aggregate_memory.merge(memory_counts)
        aggregate_candidates.merge(candidate_scores)
        folds.append(
            {
                "holdout_group": group,
                "train_documents": len(train_stems),
                "holdout_documents": len(holdout_stems),
                "memory_rows": memory_stats["row_count"],
                "baseline": baseline_counts.to_dict(),
                "with_memory": memory_counts.to_dict(),
                "candidate_memory": candidate_scores.to_dict(),
            }
        )
    return {
        "strategy": "leave_one_document_structure_group_out",
        "groups": {name: len(stems) for name, stems in sorted(groups.items())},
        "folds": folds,
        "aggregate": {
            "baseline": aggregate_baseline.to_dict(),
            "with_memory": aggregate_memory.to_dict(),
            "candidate_memory": aggregate_candidates.to_dict(),
        },
        "seconds": round(time.perf_counter() - started, 3),
    }


class SpanCounts:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, gold: Counter[Any], predicted: Counter[Any]) -> None:
        self.tp += sum((gold & predicted).values())
        self.fp += sum((predicted - gold).values())
        self.fn += sum((gold - predicted).values())

    def merge(self, other: SpanCounts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def to_dict(self) -> dict[str, float | int]:
        precision = ratio(self.tp, self.tp + self.fp)
        recall = ratio(self.tp, self.tp + self.fn)
        return {
            "true_positive": self.tp,
            "false_positive": self.fp,
            "false_negative": self.fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(ratio(2 * precision * recall, precision + recall), 6),
        }


class CandidateDecisionScores:
    def __init__(self) -> None:
        self.mentions = 0
        self.decisions = 0
        self.exact = 0
        self.weighted_numerator = 0.0
        self.weighted_denominator = 0.0

    def add(self, expected: Any, decision: tuple[str, ...] | None) -> None:
        self.mentions += 1
        if decision is None:
            return
        self.decisions += 1
        expected_set = {str(value) for value in expected}
        predicted_set = set(decision)
        self.exact += expected_set == predicted_set
        union = expected_set | predicted_set
        score = len(expected_set & predicted_set) / len(union) if union else 1.0
        weight = max(1, len(expected_set) + 1)
        self.weighted_numerator += score * weight
        self.weighted_denominator += weight

    def merge(self, other: CandidateDecisionScores) -> None:
        self.mentions += other.mentions
        self.decisions += other.decisions
        self.exact += other.exact
        self.weighted_numerator += other.weighted_numerator
        self.weighted_denominator += other.weighted_denominator

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mentions": self.mentions,
            "decisions": self.decisions,
            "coverage": round(ratio(self.decisions, self.mentions), 6),
            "exact_rate_on_decisions": round(ratio(self.exact, self.decisions), 6),
            "weighted_jaccard_on_decisions": round(
                ratio(self.weighted_numerator, self.weighted_denominator), 6
            ),
        }


def structure_group(text: str) -> str:
    case_ids = {chunk.section.split(":", 1)[0] for chunk in split_chunks(text)}
    count = len(case_ids)
    if count <= 1:
        return "single_case"
    if count >= 5:
        return "many_cases"
    return "few_cases"


def span_counter(spans: Any) -> Counter[Any]:
    return Counter(((span.start, span.end), span.type) for span in spans)


def valid_position(position: Any) -> bool:
    return bool(
        isinstance(position, list)
        and len(position) == 2
        and all(isinstance(value, int) for value in position)
    )


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
