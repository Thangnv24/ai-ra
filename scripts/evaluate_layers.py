"""Evaluate deterministic pipeline layers against paired clinical gold files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_TYPES, ASSERTION_TYPES, CODED_TYPES
from core.schema import validate_output
from extraction.context import ContextDetector
from extraction.ner import MedicalNER
from knowledge.candidates import load_slim_candidate_index
from knowledge.ontology import OntologyIndex
from knowledge.retrieval import CandidateRetriever
from scripts.gold_workflow import compare_file, mean


@dataclass
class SpanCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(self, gold: Counter[tuple[Any, ...]], predicted: Counter[tuple[Any, ...]]) -> None:
        self.true_positive += sum((gold & predicted).values())
        self.false_positive += sum((predicted - gold).values())
        self.false_negative += sum((gold - predicted).values())

    def to_dict(self) -> dict[str, float | int]:
        precision = _ratio(self.true_positive, self.true_positive + self.false_positive)
        recall = _ratio(self.true_positive, self.true_positive + self.false_negative)
        f1 = _ratio(2.0 * precision * recall, precision + recall)
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }


@dataclass
class SetScores:
    count: int = 0
    exact: int = 0
    jaccard_sum: float = 0.0
    weighted_numerator: float = 0.0
    weighted_denominator: float = 0.0
    predicted_nonempty: int = 0

    def add(self, expected: Iterable[str], predicted: Iterable[str], *, weighted: bool = False) -> None:
        expected_set = {str(item) for item in expected}
        predicted_set = {str(item) for item in predicted}
        score = _jaccard(expected_set, predicted_set)
        weight = max(1, len(expected_set) + 1) if weighted else 1
        self.count += 1
        self.exact += expected_set == predicted_set
        self.jaccard_sum += score
        self.weighted_numerator += score * weight
        self.weighted_denominator += weight
        self.predicted_nonempty += bool(predicted_set)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mentions": self.count,
            "exact_set_rate": round(_ratio(self.exact, self.count), 6),
            "mean_jaccard": round(_ratio(self.jaccard_sum, self.count), 6),
            "weighted_jaccard": round(
                _ratio(self.weighted_numerator, self.weighted_denominator), 6
            ),
            "predicted_nonempty_rate": round(_ratio(self.predicted_nonempty, self.count), 6),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure extraction, assertion, candidate, and optional end-to-end layers."
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input_part2" / "input" / "input")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "data" / "candidates")
    parser.add_argument("--lexicon-path", type=Path, default=ROOT / "data" / "external" / "vietnamese_clinical_lexicon.csv")
    parser.add_argument(
        "--with-candidates",
        action="store_true",
        help="Load the large local ICD-10/RxNorm index and run oracle-span candidate scoring.",
    )
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    report = evaluate_layers(
        input_dir=args.input_dir,
        gold_dir=args.gold_dir,
        prediction_dir=args.prediction_dir,
        candidate_dir=args.candidate_dir,
        lexicon_path=args.lexicon_path,
        with_candidates=args.with_candidates,
        limit_files=args.limit_files,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def evaluate_layers(
    *,
    input_dir: Path,
    gold_dir: Path,
    prediction_dir: Path | None = None,
    candidate_dir: Path | None = None,
    lexicon_path: Path | None = None,
    with_candidates: bool = False,
    limit_files: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_dir = input_dir.resolve()
    gold_dir = gold_dir.resolve()
    prediction_dir = prediction_dir.resolve() if prediction_dir else None
    gold_files = sorted(gold_dir.glob("*.json"), key=_natural_key)
    if limit_files is not None:
        gold_files = gold_files[: max(0, limit_files)]

    lexicon_paths = (lexicon_path.resolve(),) if lexicon_path and lexicon_path.exists() else ()
    ner = MedicalNER(lexicon_paths)
    context = ContextDetector()
    retriever: CandidateRetriever | None = None
    candidate_load_seconds = 0.0
    if with_candidates:
        candidate_started = time.perf_counter()
        slim_index = load_slim_candidate_index((candidate_dir or ROOT / "data" / "candidates").resolve())
        retriever = CandidateRetriever(OntologyIndex(()), slim_index)
        candidate_load_seconds = time.perf_counter() - candidate_started

    exact_spans = SpanCounts()
    exact_spans_by_type = {concept_type: SpanCounts() for concept_type in ALLOWED_TYPES}
    boundaries = SpanCounts()
    assertions = SetScores()
    assertions_by_type: dict[str, SetScores] = defaultdict(SetScores)
    candidates = SetScores()
    candidates_by_type: dict[str, SetScores] = defaultdict(SetScores)
    validation = {
        "missing_input_files": 0,
        "gold_files_with_strict_schema_errors": 0,
        "gold_strict_schema_errors": 0,
        "gold_offset_mismatches": 0,
        "missing_prediction_files": 0,
        "prediction_files_with_strict_schema_errors": 0,
        "prediction_strict_schema_errors": 0,
        "prediction_offset_mismatches": 0,
    }
    end_to_end_reports: list[dict[str, Any]] = []

    for gold_path in gold_files:
        input_path = input_dir / f"{gold_path.stem}.txt"
        if not input_path.exists():
            validation["missing_input_files"] += 1
            continue
        source_text = input_path.read_text(encoding="utf-8")
        gold = _load_concepts(gold_path)
        gold_errors = validate_output(gold, source_text=source_text)
        if gold_errors:
            validation["gold_files_with_strict_schema_errors"] += 1
            validation["gold_strict_schema_errors"] += len(gold_errors)
        validation["gold_offset_mismatches"] += sum(
            not _offset_matches(source_text, item) for item in gold
        )

        predicted_spans = ner.extract(source_text)
        gold_exact = Counter(_span_key(item) for item in gold if _valid_position(item))
        pred_exact = Counter((span.start, span.end, span.type) for span in predicted_spans)
        exact_spans.add(gold_exact, pred_exact)
        boundaries.add(
            Counter((start, end) for start, end, _ in gold_exact.elements()),
            Counter((start, end) for start, end, _ in pred_exact.elements()),
        )
        for concept_type in ALLOWED_TYPES:
            exact_spans_by_type[concept_type].add(
                Counter(key for key in gold_exact.elements() if key[2] == concept_type),
                Counter(key for key in pred_exact.elements() if key[2] == concept_type),
            )

        for item in gold:
            if not _valid_position(item):
                continue
            concept_type = str(item.get("type") or "")
            start, end = item["position"]
            if concept_type in ASSERTION_TYPES:
                predicted_assertions = context.assertions_for(source_text, start, end, concept_type)
                expected_assertions = item.get("assertions") or []
                assertions.add(expected_assertions, predicted_assertions)
                assertions_by_type[concept_type].add(expected_assertions, predicted_assertions)
            if retriever is not None and concept_type in CODED_TYPES:
                predicted_candidates = retriever.candidates_for(
                    str(item.get("text") or ""),
                    concept_type,
                    source_text=source_text,
                    start=start,
                    end=end,
                )
                expected_candidates = item.get("candidates") or []
                candidates.add(expected_candidates, predicted_candidates, weighted=True)
                candidates_by_type[concept_type].add(
                    expected_candidates, predicted_candidates, weighted=True
                )

        if prediction_dir is not None:
            prediction_path = prediction_dir / gold_path.name
            if prediction_path.exists():
                prediction = _load_concepts(prediction_path)
                prediction_errors = validate_output(prediction, source_text=source_text)
                if prediction_errors:
                    validation["prediction_files_with_strict_schema_errors"] += 1
                    validation["prediction_strict_schema_errors"] += len(prediction_errors)
                validation["prediction_offset_mismatches"] += sum(
                    not _offset_matches(source_text, item) for item in prediction
                )
            else:
                validation["missing_prediction_files"] += 1
                prediction = []
                prediction_errors = ["missing prediction file"]
            file_report, _ = compare_file(
                gold_path.name,
                source_text,
                gold,
                prediction,
                {"gold": gold_errors, "prediction": prediction_errors},
            )
            end_to_end_reports.append(file_report)

    boundary_summary = boundaries.to_dict()
    exact_summary = exact_spans.to_dict()
    matched_boundaries = int(boundary_summary["true_positive"])
    correctly_typed = int(exact_summary["true_positive"])
    report: dict[str, Any] = {
        "files": len(gold_files),
        "input_dir": str(input_dir),
        "gold_dir": str(gold_dir),
        "external_lexicon": str(lexicon_paths[0]) if lexicon_paths else None,
        "validation": validation,
        "rule_proposal": {
            "exact_span_and_type": exact_summary,
            "boundary_only": boundary_summary,
            "type_accuracy_on_matched_boundaries": round(
                _ratio(correctly_typed, matched_boundaries), 6
            ),
            "by_type": {
                concept_type: exact_spans_by_type[concept_type].to_dict()
                for concept_type in ALLOWED_TYPES
            },
        },
        "assertion_oracle_span": {
            **assertions.to_dict(),
            "by_type": {
                concept_type: score.to_dict()
                for concept_type, score in sorted(assertions_by_type.items())
            },
        },
        "timing": {
            "candidate_index_load_seconds": round(candidate_load_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
    }
    if retriever is not None:
        report["candidate_oracle_span"] = {
            **candidates.to_dict(),
            "by_type": {
                concept_type: score.to_dict()
                for concept_type, score in sorted(candidates_by_type.items())
            },
        }
    if prediction_dir is not None:
        report["prediction_dir"] = str(prediction_dir)
        report["end_to_end"] = _summarize_end_to_end(end_to_end_reports)
        report["end_to_end"]["comparable_to_gold"] = bool(
            validation["missing_prediction_files"] == 0
            and validation["prediction_offset_mismatches"] == 0
        )
    return report


def _summarize_end_to_end(file_reports: list[dict[str, Any]]) -> dict[str, float | int]:
    text_score = mean([report["text_score"] for report in file_reports])
    assertion_score = mean([report["assertion_score"] for report in file_reports])
    candidate_numerator = sum(report["candidate_numerator"] for report in file_reports)
    candidate_denominator = sum(report["candidate_denominator"] for report in file_reports)
    candidate_score = (
        candidate_numerator / candidate_denominator if candidate_denominator else 1.0
    )
    return {
        "files": len(file_reports),
        "text_score": round(text_score, 6),
        "assertion_score": round(assertion_score, 6),
        "candidate_score": round(candidate_score, 6),
        "final_score_estimate": round(
            100.0 * (0.3 * text_score + 0.3 * assertion_score + 0.4 * candidate_score), 4
        ),
    }


def _load_concepts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def _valid_position(item: dict[str, Any]) -> bool:
    position = item.get("position")
    return bool(
        isinstance(position, list)
        and len(position) == 2
        and all(isinstance(value, int) for value in position)
        and 0 <= position[0] <= position[1]
    )


def _offset_matches(source_text: str, item: dict[str, Any]) -> bool:
    if not _valid_position(item):
        return False
    start, end = item["position"]
    return end <= len(source_text) and source_text[start:end] == item.get("text")


def _span_key(item: dict[str, Any]) -> tuple[int, int, str]:
    start, end = item["position"]
    return start, end, str(item.get("type") or "")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return _ratio(len(left & right), len(union)) if union else 1.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (sys.maxsize, path.name)


if __name__ == "__main__":
    raise SystemExit(main())
