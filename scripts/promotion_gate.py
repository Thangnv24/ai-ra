"""Block inference changes that regress a reviewed evaluation suite."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from core.io import read_text  # noqa: E402
from core.schema import validate_output  # noqa: E402
from gold_workflow import compare_file, load_concept_list, mean, natural_key  # noqa: E402


@dataclass(frozen=True, slots=True)
class SuiteMetrics:
    files: int
    concepts: int
    text_score: float
    assertion_score: float
    candidate_score: float
    final_score: float
    validation_errors: int
    missing_files: tuple[str, ...]
    extra_files: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    incomplete_marker: bool


@dataclass(frozen=True, slots=True)
class PromotionResult:
    passed: bool
    failures: tuple[str, ...]
    prediction: SuiteMetrics
    baseline: SuiteMetrics | None


def evaluate_folder(input_dir: Path, gold_dir: Path, prediction_dir: Path) -> SuiteMetrics:
    gold_files = sorted(
        (path for path in gold_dir.glob("*.json") if path.name != "manifest.json"),
        key=natural_key,
    )
    expected_names = {path.name for path in gold_files}
    actual_names = {path.name for path in prediction_dir.glob("*.json")} if prediction_dir.exists() else set()
    missing_files = tuple(sorted(expected_names - actual_names, key=natural_name_key))
    extra_files = tuple(sorted(actual_names - expected_names, key=natural_name_key))
    missing_inputs: list[str] = []
    validation_errors = 0
    concepts = 0
    text_scores: list[float] = []
    assertion_scores: list[float] = []
    candidate_num = 0.0
    candidate_den = 0.0

    for gold_path in gold_files:
        input_path = input_dir / f"{gold_path.stem}.txt"
        pred_path = prediction_dir / gold_path.name
        if not input_path.exists():
            missing_inputs.append(input_path.name)
        source_text = read_text(input_path) if input_path.exists() else ""
        gold = load_concept_list(gold_path)
        try:
            pred = load_concept_list(pred_path) if pred_path.exists() else []
        except (json.JSONDecodeError, OSError, ValueError):
            pred = []
            validation_errors += 1
        validation = validate_output(pred, source_text=source_text or None) if pred_path.exists() else ["missing prediction file"]
        validation_errors += len(validation)
        concepts += len(pred)
        file_report, _ = compare_file(
            gold_path.name,
            source_text,
            gold,
            pred,
            {"gold": [], "prediction": validation},
        )
        text_scores.append(file_report["text_score"])
        assertion_scores.append(file_report["assertion_score"])
        candidate_num += file_report["candidate_numerator"]
        candidate_den += file_report["candidate_denominator"]

    text_score = 100.0 * mean(text_scores)
    assertion_score = 100.0 * mean(assertion_scores)
    candidate_score = 100.0 * (candidate_num / candidate_den if candidate_den else 1.0)
    final_score = 0.3 * text_score + 0.3 * assertion_score + 0.4 * candidate_score
    return SuiteMetrics(
        files=len(gold_files),
        concepts=concepts,
        text_score=round(text_score, 4),
        assertion_score=round(assertion_score, 4),
        candidate_score=round(candidate_score, 4),
        final_score=round(final_score, 4),
        validation_errors=validation_errors,
        missing_files=missing_files,
        extra_files=extra_files,
        missing_inputs=tuple(sorted(missing_inputs, key=natural_name_key)),
        incomplete_marker=(prediction_dir / "_INCOMPLETE_RUN.txt").exists(),
    )


def check_promotion(
    prediction: SuiteMetrics,
    baseline: SuiteMetrics | None,
    *,
    metric_tolerance: float = 0.5,
    final_tolerance: float = 0.25,
    min_count_ratio: float = 0.70,
    max_count_ratio: float = 1.35,
    minimums: dict[str, float] | None = None,
) -> PromotionResult:
    failures: list[str] = []
    if prediction.incomplete_marker:
        failures.append("prediction folder has _INCOMPLETE_RUN.txt")
    if prediction.missing_files:
        failures.append(f"missing prediction files: {', '.join(prediction.missing_files[:10])}")
    if prediction.extra_files:
        failures.append(f"unexpected prediction files: {', '.join(prediction.extra_files[:10])}")
    if prediction.missing_inputs:
        failures.append(f"missing input files: {', '.join(prediction.missing_inputs[:10])}")
    if prediction.validation_errors:
        failures.append(f"schema/offset validation errors: {prediction.validation_errors}")

    reference_count = baseline.concepts if baseline is not None else 0
    if reference_count:
        ratio = prediction.concepts / reference_count
        if ratio < min_count_ratio or ratio > max_count_ratio:
            failures.append(
                f"concept count ratio {ratio:.3f} outside [{min_count_ratio:.3f}, {max_count_ratio:.3f}]"
            )

    if baseline is not None:
        if baseline.incomplete_marker:
            failures.append("baseline folder has _INCOMPLETE_RUN.txt")
        if baseline.missing_files or baseline.extra_files or baseline.missing_inputs:
            failures.append("baseline folder is structurally incomplete or mixed")
        if baseline.validation_errors:
            failures.append(f"baseline schema/offset validation errors: {baseline.validation_errors}")
        for name in ("text_score", "assertion_score", "candidate_score"):
            current = getattr(prediction, name)
            previous = getattr(baseline, name)
            if current + metric_tolerance < previous:
                failures.append(f"{name} regressed {previous:.4f} -> {current:.4f}")
        if prediction.final_score + final_tolerance < baseline.final_score:
            failures.append(
                f"final_score regressed {baseline.final_score:.4f} -> {prediction.final_score:.4f}"
            )

    for name, minimum in (minimums or {}).items():
        current = getattr(prediction, name)
        if current < minimum:
            failures.append(f"{name} {current:.4f} is below minimum {minimum:.4f}")

    return PromotionResult(not failures, tuple(failures), prediction, baseline)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and compare an inference run before promotion.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--metric-tolerance", type=float, default=0.5)
    parser.add_argument("--final-tolerance", type=float, default=0.25)
    parser.add_argument("--min-count-ratio", type=float, default=0.70)
    parser.add_argument("--max-count-ratio", type=float, default=1.35)
    parser.add_argument("--min-text-score", type=float, default=0.0)
    parser.add_argument("--min-assertion-score", type=float, default=0.0)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    prediction = evaluate_folder(args.input_dir.resolve(), args.gold_dir.resolve(), args.prediction_dir.resolve())
    baseline = None
    if args.baseline_dir:
        baseline = evaluate_folder(args.input_dir.resolve(), args.gold_dir.resolve(), args.baseline_dir.resolve())
    result = check_promotion(
        prediction,
        baseline,
        metric_tolerance=args.metric_tolerance,
        final_tolerance=args.final_tolerance,
        min_count_ratio=args.min_count_ratio,
        max_count_ratio=args.max_count_ratio,
        minimums={
            "text_score": args.min_text_score,
            "assertion_score": args.min_assertion_score,
            "candidate_score": args.min_candidate_score,
        },
    )
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "passed": result.passed,
        "failures": list(result.failures),
        "prediction": asdict(result.prediction),
        "baseline": asdict(result.baseline) if result.baseline is not None else None,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.passed else 1


def natural_name_key(name: str) -> list[int | str]:
    import re

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)]


if __name__ == "__main__":
    raise SystemExit(main())
