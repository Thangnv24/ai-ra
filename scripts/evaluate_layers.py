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
from extraction.annotation_memory import AnnotationMemory
from extraction.assertion_model import AssertionClassifier
from extraction.context import ContextDetector
from extraction.learned_models import SpanAcceptanceModel, TokenSpanModel
from extraction.ner import MedicalNER
from extraction.span_grammar import SpanGrammar
from extraction.span_verifier import SpanTypeVerifier
from knowledge.candidates import load_slim_candidate_index
from knowledge.ontology import OntologyIndex
from knowledge.retrieval import CandidateQueryAliases, CandidateRetriever
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
    parser.add_argument("--lexicon-path", type=Path)
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=ROOT / "data" / "external" / "annotation_memory.jsonl",
    )
    parser.add_argument(
        "--assertion-model-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--token-model-path",
        type=Path,
        default=ROOT / "data" / "external" / "token_span_model.json.gz",
    )
    parser.add_argument(
        "--acceptance-model-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--span-grammar-path",
        type=Path,
        default=ROOT / "data" / "external" / "span_grammar.json",
    )
    parser.add_argument(
        "--candidate-alias-path",
        type=Path,
        default=ROOT / "data" / "external" / "candidate_query_aliases.json",
    )
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
        memory_path=args.memory_path,
        token_model_path=args.token_model_path,
        acceptance_model_path=args.acceptance_model_path,
        assertion_model_path=args.assertion_model_path,
        span_grammar_path=args.span_grammar_path,
        candidate_alias_path=args.candidate_alias_path,
        with_candidates=args.with_candidates,
        limit_files=args.limit_files,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    try:
        print(payload)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(payload.encode("utf-8") + b"\n")
    return 0


def evaluate_layers(
    *,
    input_dir: Path,
    gold_dir: Path,
    prediction_dir: Path | None = None,
    candidate_dir: Path | None = None,
    lexicon_path: Path | None = None,
    memory_path: Path | None = None,
    token_model_path: Path | None = None,
    acceptance_model_path: Path | None = None,
    assertion_model_path: Path | None = None,
    span_grammar_path: Path | None = None,
    candidate_alias_path: Path | None = None,
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
    memory = AnnotationMemory.load(memory_path.resolve()) if memory_path and memory_path.exists() else AnnotationMemory.empty()
    token_model = (
        TokenSpanModel.load(token_model_path.resolve())
        if token_model_path and token_model_path.exists()
        else TokenSpanModel.empty()
    )
    acceptance_model = (
        SpanAcceptanceModel.load(acceptance_model_path.resolve())
        if acceptance_model_path and acceptance_model_path.exists()
        else SpanAcceptanceModel.empty()
    )
    verifier = SpanTypeVerifier(memory, acceptance_model)
    grammar_path = span_grammar_path or (ROOT / "data" / "external" / "span_grammar.json")
    span_grammar = SpanGrammar.load(grammar_path.resolve())
    assertion_model = (
        AssertionClassifier.load(assertion_model_path.resolve())
        if assertion_model_path and assertion_model_path.exists()
        else AssertionClassifier.empty()
    )
    context = ContextDetector(assertion_model)
    retriever: CandidateRetriever | None = None
    candidate_load_seconds = 0.0
    if with_candidates:
        candidate_started = time.perf_counter()
        slim_index = load_slim_candidate_index((candidate_dir or ROOT / "data" / "candidates").resolve())
        alias_path = candidate_alias_path or (ROOT / "data" / "external" / "candidate_query_aliases.json")
        query_aliases = CandidateQueryAliases.load(alias_path.resolve())
        retriever = CandidateRetriever(OntologyIndex(()), slim_index, memory, query_aliases)
        candidate_load_seconds = time.perf_counter() - candidate_started

    exact_spans = SpanCounts()
    exact_spans_by_type = {concept_type: SpanCounts() for concept_type in ALLOWED_TYPES}
    boundaries = SpanCounts()
    verified_spans = SpanCounts()
    verified_spans_by_type = {concept_type: SpanCounts() for concept_type in ALLOWED_TYPES}
    verified_boundaries = SpanCounts()
    lattice_spans = SpanCounts()
    lattice_boundaries = SpanCounts()
    lattice_by_variant: dict[str, SpanCounts] = defaultdict(SpanCounts)
    assertions = SetScores()
    assertions_by_type: dict[str, SetScores] = defaultdict(SetScores)
    candidates = SetScores()
    candidates_by_type: dict[str, SetScores] = defaultdict(SetScores)
    proposal_sources: dict[str, dict[str, SpanCounts]] = defaultdict(
        lambda: {"exact": SpanCounts(), "boundary": SpanCounts()}
    )
    verified_error_taxonomy: Counter[str] = Counter()
    prediction_exact = SpanCounts()
    prediction_boundaries = SpanCounts()
    prediction_by_type = {concept_type: SpanCounts() for concept_type in ALLOWED_TYPES}
    prediction_type_confusions: Counter[str] = Counter()
    prediction_error_taxonomy: Counter[str] = Counter()
    prediction_assertions = SetScores()
    prediction_assertions_by_type: dict[str, SetScores] = defaultdict(SetScores)
    prediction_candidates = SetScores()
    prediction_candidates_by_type: dict[str, SetScores] = defaultdict(SetScores)
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

        rule_proposals = ner.propose(source_text)
        memory_proposals = memory.propose(source_text)
        sequence_proposals = token_model.propose(source_text)
        predicted_spans = ner.extract(source_text)
        grammar_spans, _ = span_grammar.expand(
            source_text, [*rule_proposals, *memory_proposals]
        )
        corroborated_sequence = _corroborated_sequence_spans(sequence_proposals, grammar_spans)
        proposal_lattice = [*grammar_spans, *corroborated_sequence]
        selected_spans, _ = verifier.select(source_text, proposal_lattice)
        gold_exact = Counter(_span_key(item) for item in gold if _valid_position(item))
        gold_boundary = Counter((start, end) for start, end, _ in gold_exact.elements())
        pred_exact = Counter((span.start, span.end, span.type) for span in predicted_spans)
        selected_exact = Counter((span.start, span.end, span.type) for span in selected_spans)
        lattice_exact = Counter((span.start, span.end, span.type) for span in proposal_lattice)
        exact_spans.add(gold_exact, pred_exact)
        boundaries.add(
            Counter((start, end) for start, end, _ in gold_exact.elements()),
            Counter((start, end) for start, end, _ in pred_exact.elements()),
        )
        verified_spans.add(gold_exact, selected_exact)
        lattice_spans.add(gold_exact, lattice_exact)
        lattice_boundaries.add(
            gold_boundary,
            Counter((start, end) for start, end, _ in lattice_exact.elements()),
        )
        for variant, variant_spans in _group_spans_by_variant(proposal_lattice).items():
            lattice_by_variant[variant].add(
                gold_exact,
                Counter((span.start, span.end, span.type) for span in variant_spans),
            )
        verified_boundaries.add(
            gold_boundary,
            Counter((start, end) for start, end, _ in selected_exact.elements()),
        )
        for source, source_spans in _group_spans_by_source(
            proposal_lattice
        ).items():
            source_exact = Counter((span.start, span.end, span.type) for span in source_spans)
            proposal_sources[source]["exact"].add(gold_exact, source_exact)
            proposal_sources[source]["boundary"].add(
                gold_boundary,
                Counter((start, end) for start, end, _ in source_exact.elements()),
            )
        verified_error_taxonomy.update(_error_taxonomy(gold_exact, selected_exact))
        for concept_type in ALLOWED_TYPES:
            exact_spans_by_type[concept_type].add(
                Counter(key for key in gold_exact.elements() if key[2] == concept_type),
                Counter(key for key in pred_exact.elements() if key[2] == concept_type),
            )
            verified_spans_by_type[concept_type].add(
                Counter(key for key in gold_exact.elements() if key[2] == concept_type),
                Counter(key for key in selected_exact.elements() if key[2] == concept_type),
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
            predicted_exact = Counter(
                _span_key(item) for item in prediction if _valid_position(item)
            )
            predicted_boundary = Counter(
                (start, end) for start, end, _ in predicted_exact.elements()
            )
            prediction_exact.add(gold_exact, predicted_exact)
            prediction_boundaries.add(gold_boundary, predicted_boundary)
            prediction_error_taxonomy.update(_error_taxonomy(gold_exact, predicted_exact))
            prediction_type_confusions.update(_type_confusions(gold_exact, predicted_exact))
            for concept_type in ALLOWED_TYPES:
                prediction_by_type[concept_type].add(
                    Counter(key for key in gold_exact.elements() if key[2] == concept_type),
                    Counter(key for key in predicted_exact.elements() if key[2] == concept_type),
                )
            _score_conditional_fields(
                gold,
                prediction,
                prediction_assertions,
                prediction_assertions_by_type,
                prediction_candidates,
                prediction_candidates_by_type,
            )
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
        "annotation_memory": str(memory_path.resolve()) if memory_path and memory_path.exists() else None,
        "token_span_model": (
            str(token_model_path.resolve()) if token_model_path and token_model_path.exists() else None
        ),
        "span_acceptance_model": (
            str(acceptance_model_path.resolve())
            if acceptance_model_path and acceptance_model_path.exists()
            else None
        ),
        "span_grammar": str(grammar_path.resolve()),
        "assertion_model": (
            str(assertion_model_path.resolve())
            if assertion_model_path and assertion_model_path.exists()
            else None
        ),
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
        "verified_proposal": {
            "exact_span_and_type": verified_spans.to_dict(),
            "boundary_only": verified_boundaries.to_dict(),
            "by_type": {
                concept_type: verified_spans_by_type[concept_type].to_dict()
                for concept_type in ALLOWED_TYPES
            },
            "error_taxonomy": dict(sorted(verified_error_taxonomy.items())),
        },
        "proposal_lattice": {
            "exact_span_and_type": lattice_spans.to_dict(),
            "boundary_only": lattice_boundaries.to_dict(),
            "by_variant": {
                variant: scores.to_dict()
                for variant, scores in sorted(lattice_by_variant.items())
            },
        },
        "proposal_sources": {
            source: {
                "exact_span_and_type": scores["exact"].to_dict(),
                "boundary_only": scores["boundary"].to_dict(),
            }
            for source, scores in sorted(proposal_sources.items())
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
        prediction_boundary_summary = prediction_boundaries.to_dict()
        prediction_exact_summary = prediction_exact.to_dict()
        report["prediction_diagnostics"] = {
            "exact_span_and_type": prediction_exact_summary,
            "boundary_only": prediction_boundary_summary,
            "type_accuracy_on_matched_boundaries": round(
                _ratio(
                    int(prediction_exact_summary["true_positive"]),
                    int(prediction_boundary_summary["true_positive"]),
                ),
                6,
            ),
            "by_type": {
                concept_type: prediction_by_type[concept_type].to_dict()
                for concept_type in ALLOWED_TYPES
            },
            "type_confusions": dict(sorted(prediction_type_confusions.items())),
            "error_taxonomy": dict(sorted(prediction_error_taxonomy.items())),
            "assertion_on_exact_spans": {
                **prediction_assertions.to_dict(),
                "by_type": {
                    concept_type: score.to_dict()
                    for concept_type, score in sorted(prediction_assertions_by_type.items())
                },
            },
            "candidate_on_exact_spans": {
                **prediction_candidates.to_dict(),
                "by_type": {
                    concept_type: score.to_dict()
                    for concept_type, score in sorted(prediction_candidates_by_type.items())
                },
            },
        }
    return report


def _group_spans_by_source(spans: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for span in spans:
        grouped[str(getattr(span, "source", "unknown") or "unknown")].append(span)
    return grouped


def _corroborated_sequence_spans(sequence_spans: Iterable[Any], trusted_spans: Iterable[Any]) -> list[Any]:
    trusted_keys = {(span.start, span.end, span.type) for span in trusted_spans}
    return [
        span
        for span in sequence_spans
        if (span.start, span.end, span.type) in trusted_keys
    ]


def _group_spans_by_variant(spans: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for span in spans:
        variant = str(getattr(span, "variant", "original") or "original")
        grouped[variant].append(span)
    return grouped


def _error_taxonomy(
    gold: Counter[tuple[int, int, str]],
    predicted: Counter[tuple[int, int, str]],
) -> Counter[str]:
    errors: Counter[str] = Counter()
    gold_items = list((gold - predicted).elements())
    for pred_start, pred_end, pred_type in (predicted - gold).elements():
        same_boundary = [item for item in gold_items if item[:2] == (pred_start, pred_end)]
        if same_boundary:
            errors["wrong_type"] += 1
            continue
        overlaps = [
            item
            for item in gold_items
            if pred_start < item[1] and item[0] < pred_end
        ]
        same_type = [item for item in overlaps if item[2] == pred_type]
        candidates = same_type or overlaps
        if not candidates:
            errors["no_gold_overlap"] += 1
            continue
        gold_start, gold_end, _ = max(
            candidates,
            key=lambda item: min(pred_end, item[1]) - max(pred_start, item[0]),
        )
        if gold_start <= pred_start and pred_end <= gold_end:
            errors["boundary_too_short"] += 1
        elif pred_start <= gold_start and gold_end <= pred_end:
            errors["boundary_too_long"] += 1
        else:
            errors["boundary_shift"] += 1
    return errors


def _type_confusions(
    gold: Counter[tuple[int, int, str]],
    predicted: Counter[tuple[int, int, str]],
) -> Counter[str]:
    gold_by_boundary: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    predicted_by_boundary: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for start, end, concept_type in gold.elements():
        gold_by_boundary[(start, end)][concept_type] += 1
    for start, end, concept_type in predicted.elements():
        predicted_by_boundary[(start, end)][concept_type] += 1
    output: Counter[str] = Counter()
    for boundary in gold_by_boundary.keys() & predicted_by_boundary.keys():
        missing = gold_by_boundary[boundary] - predicted_by_boundary[boundary]
        extra = predicted_by_boundary[boundary] - gold_by_boundary[boundary]
        for gold_type, gold_count in missing.items():
            for predicted_type, predicted_count in extra.items():
                count = min(gold_count, predicted_count)
                if count:
                    output[f"{gold_type} -> {predicted_type}"] += count
    return output


def _score_conditional_fields(
    gold: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    assertions: SetScores,
    assertions_by_type: dict[str, SetScores],
    candidates: SetScores,
    candidates_by_type: dict[str, SetScores],
) -> None:
    gold_by_span: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    predicted_by_span: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in gold:
        if _valid_position(item):
            gold_by_span[_span_key(item)].append(item)
    for item in predicted:
        if _valid_position(item):
            predicted_by_span[_span_key(item)].append(item)
    for key in gold_by_span.keys() & predicted_by_span.keys():
        concept_type = key[2]
        for expected, actual in zip(gold_by_span[key], predicted_by_span[key]):
            if concept_type in ASSERTION_TYPES:
                assertions.add(expected.get("assertions") or [], actual.get("assertions") or [])
                assertions_by_type[concept_type].add(
                    expected.get("assertions") or [], actual.get("assertions") or []
                )
            if concept_type in CODED_TYPES:
                candidates.add(
                    expected.get("candidates") or [],
                    actual.get("candidates") or [],
                    weighted=True,
                )
                candidates_by_type[concept_type].add(
                    expected.get("candidates") or [],
                    actual.get("candidates") or [],
                    weighted=True,
                )


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
