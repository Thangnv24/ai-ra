"""Train and cross-validate the post-span assertion classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_ASSERTIONS, ASSERTION_TYPES
from extraction.assertion_model import assertion_features
from extraction.context import ContextDetector
from scripts.train_span_models import grouped_folds, load_documents, train_logistic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train assertion classifier with grouped OOF scoring.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input_part2" / "input" / "input")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "external" / "assertion_model.json")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "external" / "assertion_model_cv_report.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "external" / "assertion_model_manifest.json")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    documents = load_documents(args.input_dir.resolve(), args.gold_dir.resolve())
    folds = grouped_folds(documents, max(2, args.folds))
    examples = build_examples(documents)
    document_fold = {
        document_id: fold_index
        for fold_index, fold in enumerate(folds)
        for document_id in fold
    }
    oof_probabilities: dict[str, list[float]] = {
        label: [0.0] * len(examples) for label in ALLOWED_ASSERTIONS
    }
    models: dict[str, dict[str, Any]] = {}
    for label in ALLOWED_ASSERTIONS:
        for fold_index in range(len(folds)):
            train = _label_examples(
                examples,
                label,
                lambda item: document_fold[item["document_id"]] != fold_index,
            )
            bias, weights = train_logistic(train, epochs=7, max_positive_weight=10.0)
            for index, item in enumerate(examples):
                if document_fold[item["document_id"]] != fold_index:
                    continue
                oof_probabilities[label][index] = _probability(bias, weights, item["features"])
        threshold = tune_threshold(examples, oof_probabilities[label], label)
        final_examples = _label_examples(examples, label, lambda item: True)
        bias, weights = train_logistic(final_examples, epochs=10, max_positive_weight=10.0)
        models[label] = {
            "bias": round(bias, 8),
            "weights": {key: round(value, 8) for key, value in sorted(weights.items())},
            "threshold": threshold,
        }

    report = evaluate_oof(examples, oof_probabilities, models)
    runtime_models = {
        label: model for label, model in models.items() if label != "isFamily"
    }
    payload = {"format_version": 1, "models": runtime_models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "format_version": 1,
        "source_documents": len(documents),
        "examples": len(examples),
        "features": {label: len(model["weights"]) for label, model in runtime_models.items()},
        "artifact": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "report": str(args.report.resolve()),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "report": report}, ensure_ascii=False, indent=2))
    return 0


def build_examples(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detector = ContextDetector()
    output: list[dict[str, Any]] = []
    for document_id, document in enumerate(documents):
        text = str(document["text"])
        for item in document["gold"]:
            concept_type = str(item["type"])
            if concept_type not in ASSERTION_TYPES:
                continue
            start, end = item["position"]
            rules = detector.assertions_for(text, start, end, concept_type)
            output.append(
                {
                    "document_id": document_id,
                    "features": assertion_features(text, start, end, concept_type, rules),
                    "expected": set(str(value) for value in item.get("assertions") or ()),
                    "rules": set(rules),
                }
            )
    return output


def tune_threshold(examples: list[dict[str, Any]], scores: list[float], label: str) -> float:
    beta = 0.8 if label == "isNegated" else 1.0
    if label == "isFamily":
        beta = 0.5
    beta2 = beta * beta
    def objective(threshold: float) -> float:
        tp = fp = fn = 0
        for item, score in zip(examples, scores):
            predicted = score >= threshold
            expected = label in item["expected"]
            tp += predicted and expected
            fp += predicted and not expected
            fn += not predicted and expected
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return (1 + beta2) * precision * recall / max(1e-9, beta2 * precision + recall)
    candidates = [value / 100.0 for value in range(10, 96, 2)]
    return round(max(candidates, key=objective), 2)


def evaluate_oof(
    examples: list[dict[str, Any]],
    probabilities: dict[str, list[float]],
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exact = 0
    label_counts: dict[str, dict[str, int | float]] = {}
    for label in ALLOWED_ASSERTIONS:
        threshold = float(models[label]["threshold"])
        tp = fp = fn = 0
        for index, item in enumerate(examples):
            predicted = probabilities[label][index] >= threshold
            expected = label in item["expected"]
            tp += predicted and expected
            fp += predicted and not expected
            fn += not predicted and expected
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        label_counts[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2 * precision * recall / max(1e-9, precision + recall), 6),
            "threshold": threshold,
        }
    for index, item in enumerate(examples):
        selected: set[str] = set()
        for label in ALLOWED_ASSERTIONS:
            probability = probabilities[label][index]
            threshold = float(models[label]["threshold"])
            if probability >= threshold:
                selected.add(label)
            elif label in item["rules"] and probability >= max(0.05, threshold * 0.55):
                selected.add(label)
        exact += selected == item["expected"]
    return {
        "mentions": len(examples),
        "exact_set_rate": round(exact / max(1, len(examples)), 6),
        "by_label": label_counts,
    }


def _label_examples(examples, label, predicate):
    return [
        {
            "features": item["features"],
            "label": int(label in item["expected"]),
        }
        for item in examples
        if predicate(item)
    ]


def _probability(bias: float, weights: dict[str, float], features) -> float:
    logit = bias + sum(weights.get(feature, 0.0) for feature in features)
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
