from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ASSERTION_TYPES, CODED_TYPES
from core.io import load_json, read_text
from core.schema import validate_output
from core.text import normalize_key


@dataclass(frozen=True)
class Match:
    gold_index: int
    pred_index: int
    score: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual gold draft, scoring, and error-report workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create editable manual-gold draft from a prediction folder.")
    init_parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    init_parser.add_argument("--prediction-dir", type=Path, required=True)
    init_parser.add_argument("--gold-dir", type=Path, default=None)
    init_parser.add_argument("--pretty", action="store_true")

    score_parser = subparsers.add_parser("score", help="Score a prediction folder against manual gold JSON files.")
    score_parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    score_parser.add_argument("--gold-dir", type=Path, required=True)
    score_parser.add_argument("--prediction-dir", type=Path, required=True)
    score_parser.add_argument("--report", type=Path, default=None)
    score_parser.add_argument("--errors", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "init":
        return init_gold_draft(args)
    if args.command == "score":
        return score_predictions(args)
    raise AssertionError(args.command)


def init_gold_draft(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()
    if not prediction_dir.exists():
        raise FileNotFoundError(f"prediction dir not found: {prediction_dir}")
    gold_dir = (args.gold_dir or (ROOT / "data" / "gold_manual" / f"draft_{timestamp()}")).resolve()
    gold_dir.mkdir(parents=True, exist_ok=True)

    queue_path = gold_dir / "review_queue.jsonl"
    files = sorted(prediction_dir.glob("*.json"), key=natural_key)
    copied = 0
    issue_count = 0
    with queue_path.open("w", encoding="utf-8") as queue:
        for pred_path in files:
            input_path = input_dir / f"{pred_path.stem}.txt"
            source_text = read_text(input_path) if input_path.exists() else ""
            payload = load_json(pred_path)
            errors = validate_output(payload, source_text=source_text or None)
            if errors:
                issue_count += 1
            out_path = gold_dir / pred_path.name
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None),
                encoding="utf-8",
            )
            copied += 1
            for index, concept in enumerate(payload if isinstance(payload, list) else []):
                if not isinstance(concept, dict):
                    continue
                start, end = concept_position(concept)
                queue.write(
                    json.dumps(
                        {
                            "file": pred_path.name,
                            "concept_index": index,
                            "review": "check",
                            "text": concept.get("text"),
                            "type": concept.get("type"),
                            "position": concept.get("position"),
                            "assertions": concept.get("assertions", []),
                            "candidates": concept.get("candidates", []),
                            "context": context_window(source_text, start, end),
                            "validation_errors": errors[:10] if index == 0 else [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "prediction_dir": str(prediction_dir),
        "gold_dir": str(gold_dir),
        "files": copied,
        "files_with_validation_issues": issue_count,
        "next_step": "Edit the JSON files in gold_dir, then run the score command.",
    }
    (gold_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def score_predictions(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    gold_dir = args.gold_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()
    report_path = (args.report or (ROOT / "data" / "gold_manual" / f"score_report_{timestamp()}.json")).resolve()
    errors_path = (args.errors or report_path.with_suffix(".errors.jsonl")).resolve()

    file_reports: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    text_scores: list[float] = []
    assertion_scores: list[float] = []
    candidate_num = 0.0
    candidate_den = 0.0

    gold_files = sorted(gold_dir.glob("*.json"), key=natural_key)
    for gold_path in gold_files:
        if gold_path.name in {"manifest.json"}:
            continue
        pred_path = prediction_dir / gold_path.name
        input_path = input_dir / f"{gold_path.stem}.txt"
        source_text = read_text(input_path) if input_path.exists() else ""
        gold = load_concept_list(gold_path)
        pred = load_concept_list(pred_path) if pred_path.exists() else []

        validation = {
            "gold": validate_output(gold, source_text=source_text or None),
            "prediction": validate_output(pred, source_text=source_text or None) if pred_path.exists() else ["missing prediction file"],
        }
        file_report, rows = compare_file(gold_path.name, source_text, gold, pred, validation)
        file_reports.append(file_report)
        error_rows.extend(rows)
        text_scores.append(file_report["text_score"])
        assertion_scores.append(file_report["assertion_score"])
        candidate_num += file_report["candidate_numerator"]
        candidate_den += file_report["candidate_denominator"]

    text_score = mean(text_scores)
    assertion_score = mean(assertion_scores)
    candidate_score = candidate_num / candidate_den if candidate_den else 1.0
    final_score = 100.0 * (0.3 * text_score + 0.3 * assertion_score + 0.4 * candidate_score)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "gold_dir": str(gold_dir),
        "prediction_dir": str(prediction_dir),
        "num_records": len(file_reports),
        "text_score": round(text_score * 100.0, 4),
        "J_assertion": round(assertion_score * 100.0, 4),
        "J_candidates": round(candidate_score * 100.0, 4),
        "final_score_estimate": round(final_score, 4),
        "files": file_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in error_rows),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("num_records", "text_score", "J_assertion", "J_candidates", "final_score_estimate")}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    print(f"errors={errors_path}")
    return 0


def compare_file(
    file_name: str,
    source_text: str,
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    validation: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = greedy_matches(gold, pred)
    matched_gold = {match.gold_index for match in matches}
    matched_pred = {match.pred_index for match in matches}
    rows: list[dict[str, Any]] = []

    text_units: list[float] = []
    assertion_units: list[float] = []
    candidate_num = 0.0
    candidate_den = 0.0

    for match in matches:
        gold_item = gold[match.gold_index]
        pred_item = pred[match.pred_index]
        text_score = concept_text_score(str(gold_item.get("text", "")), str(pred_item.get("text", "")))
        assertion_score = jaccard(gold_item.get("assertions", []), pred_item.get("assertions", []))
        text_units.append(text_score)
        if gold_item.get("type") in ASSERTION_TYPES or pred_item.get("type") in ASSERTION_TYPES:
            assertion_units.append(assertion_score)
        if is_coded_pair(gold_item, pred_item):
            cand_score = jaccard(gold_item.get("candidates", []), pred_item.get("candidates", []))
            weight = max(1, len(as_str_set(gold_item.get("candidates", []))) + 1)
            candidate_num += cand_score * weight
            candidate_den += weight
            if cand_score < 1.0:
                rows.append(error_row(file_name, "candidate_mismatch", source_text, gold_item, pred_item, match.score))
        if text_score < 1.0:
            rows.append(error_row(file_name, "text_mismatch", source_text, gold_item, pred_item, match.score))
        if assertion_score < 1.0:
            rows.append(error_row(file_name, "assertion_mismatch", source_text, gold_item, pred_item, match.score))

    for index, gold_item in enumerate(gold):
        if index in matched_gold:
            continue
        text_units.append(0.0)
        if gold_item.get("type") in ASSERTION_TYPES:
            assertion_units.append(0.0)
        if gold_item.get("type") in CODED_TYPES:
            candidate_den += max(1, len(as_str_set(gold_item.get("candidates", []))) + 1)
        rows.append(error_row(file_name, "missing_gold", source_text, gold_item, None, 0.0))

    for index, pred_item in enumerate(pred):
        if index in matched_pred:
            continue
        text_units.append(0.0)
        if pred_item.get("type") in ASSERTION_TYPES:
            assertion_units.append(0.0)
        if pred_item.get("type") in CODED_TYPES:
            candidate_den += max(1, len(as_str_set(pred_item.get("candidates", []))) + 1)
        rows.append(error_row(file_name, "extra_prediction", source_text, None, pred_item, 0.0))

    text_score = mean(text_units)
    assertion_score = mean(assertion_units)
    candidate_score = candidate_num / candidate_den if candidate_den else 1.0
    report = {
        "file": file_name,
        "gold": len(gold),
        "prediction": len(pred),
        "matched": len(matches),
        "missing": len(gold) - len(matched_gold),
        "extra": len(pred) - len(matched_pred),
        "text_score": round(text_score, 6),
        "assertion_score": round(assertion_score, 6),
        "candidate_score": round(candidate_score, 6),
        "candidate_numerator": candidate_num,
        "candidate_denominator": candidate_den,
        "validation": validation,
    }
    return report, rows


def greedy_matches(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> list[Match]:
    candidates: list[Match] = []
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(pred):
            score = match_score(gold_item, pred_item)
            if score >= 0.25:
                candidates.append(Match(gold_index, pred_index, score))
    candidates.sort(key=lambda item: item.score, reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    selected: list[Match] = []
    for item in candidates:
        if item.gold_index in used_gold or item.pred_index in used_pred:
            continue
        used_gold.add(item.gold_index)
        used_pred.add(item.pred_index)
        selected.append(item)
    return selected


def match_score(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    if gold.get("type") != pred.get("type"):
        return 0.0
    gold_start, gold_end = concept_position(gold)
    pred_start, pred_end = concept_position(pred)
    overlap = span_overlap(gold_start, gold_end, pred_start, pred_end)
    union = max(gold_end, pred_end) - min(gold_start, pred_start)
    overlap_score = overlap / union if union > 0 else 0.0
    text_score = concept_text_score(str(gold.get("text", "")), str(pred.get("text", "")))
    return 0.7 * overlap_score + 0.3 * text_score


def concept_text_score(gold_text: str, pred_text: str) -> float:
    gold_tokens = normalize_key(gold_text).split()
    pred_tokens = normalize_key(pred_text).split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    distance = edit_distance(gold_tokens, pred_tokens)
    return max(0.0, 1.0 - distance / max(1, len(gold_tokens)))


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def error_row(
    file_name: str,
    kind: str,
    source_text: str,
    gold: dict[str, Any] | None,
    pred: dict[str, Any] | None,
    match_score_value: float,
) -> dict[str, Any]:
    concept = gold or pred or {}
    start, end = concept_position(concept)
    return {
        "file": file_name,
        "kind": kind,
        "match_score": round(match_score_value, 6),
        "gold": compact_concept(gold),
        "prediction": compact_concept(pred),
        "context": context_window(source_text, start, end),
    }


def compact_concept(concept: dict[str, Any] | None) -> dict[str, Any] | None:
    if concept is None:
        return None
    item = {
        "text": concept.get("text"),
        "type": concept.get("type"),
        "position": concept.get("position"),
        "assertions": concept.get("assertions", []),
    }
    if concept.get("type") in CODED_TYPES or concept.get("candidates") is not None:
        item["candidates"] = concept.get("candidates", [])
    return item


def load_concept_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected JSON list")
    return [item for item in payload if isinstance(item, dict)]


def concept_position(concept: dict[str, Any]) -> tuple[int, int]:
    position = concept.get("position")
    if isinstance(position, list) and len(position) == 2 and all(isinstance(value, int) for value in position):
        return position[0], position[1]
    return 0, 0


def context_window(text: str, start: int, end: int, radius: int = 120) -> str:
    if not text:
        return ""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].replace("\r", " ").replace("\n", " ")


def jaccard(left: Any, right: Any) -> float:
    left_set = as_str_set(left)
    right_set = as_str_set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def as_str_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if isinstance(item, str) and item}


def is_coded_pair(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    return gold.get("type") in CODED_TYPES or pred.get("type") in CODED_TYPES


def span_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def natural_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem) if stem.isdigit() else 10**9, path.name)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
