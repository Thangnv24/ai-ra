"""Train dependency-free token proposal and span acceptance models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_TYPES
from core.text import normalize_key
from extraction.annotation_memory import AnnotationMemory, section_at
from extraction.boundary_variants import BoundaryVariantGenerator
from extraction.learned_models import (
    AveragedPerceptronTrainer,
    SpanAcceptanceModel,
    Token,
    TokenSpanModel,
    span_features,
    tokenize,
)
from extraction.ner import MedicalNER, SpanCandidate
from extraction.sectioning import detect_sections, detect_subsections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train token proposal and span acceptance artifacts.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input_part2" / "input" / "input")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument("--memory", type=Path, default=ROOT / "data" / "external" / "annotation_memory.jsonl")
    parser.add_argument("--archived-proposals", type=Path, default=ROOT / "output" / "2" / "part2")
    parser.add_argument("--token-model", type=Path, default=ROOT / "data" / "external" / "token_span_model.json.gz")
    parser.add_argument("--acceptance-model", type=Path, default=ROOT / "data" / "external" / "span_acceptance_model.json")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "external" / "span_model_cv_report.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "external" / "span_model_manifest.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--reuse-token-model",
        action="store_true",
        help="Reuse an existing token model and skip its expensive OOF retraining.",
    )
    args = parser.parse_args(argv)

    documents = load_documents(args.input_dir.resolve(), args.gold_dir.resolve())
    folds = grouped_folds(documents, max(2, args.folds))
    labels = ("O",) + tuple(
        label
        for concept_type in ALLOWED_TYPES
        for label in (f"B:{concept_type}", f"I:{concept_type}")
    )

    if args.reuse_token_model and args.token_model.exists():
        token_oof: dict[str, Any] = {"status": "reused_existing_artifact"}
        final_token_model = TokenSpanModel.load(args.token_model)
    else:
        token_oof = evaluate_token_oof(documents, folds, labels)
        final_token_model = train_token_model(documents, labels)
        final_token_model.save(args.token_model)
    print(json.dumps({"stage": "token_model", "result": token_oof}), flush=True)

    memory = AnnotationMemory.load(args.memory.resolve())
    examples, proposal_stats = build_span_examples(
        documents,
        final_token_model,
        memory,
        args.archived_proposals.resolve() if args.archived_proposals.exists() else None,
    )
    examples = compact_training_examples(examples, negative_ratio=4)
    proposal_stats["examples_after_compaction"] = len(examples)
    proposal_stats["positive_after_compaction"] = sum(item["label"] for item in examples)
    proposal_stats["negative_after_compaction"] = sum(not item["label"] for item in examples)
    print(json.dumps({"stage": "span_examples", "result": proposal_stats}), flush=True)
    oof_scores = acceptance_oof(examples, folds)
    thresholds = tune_thresholds(examples, oof_scores)
    bias, weights = train_logistic(examples, epochs=9)
    acceptance_payload = {
        "format_version": 1,
        "bias": round(bias, 8),
        "weights": {key: round(value, 8) for key, value in sorted(weights.items())},
        "thresholds": thresholds,
    }
    args.acceptance_model.parent.mkdir(parents=True, exist_ok=True)
    args.acceptance_model.write_text(
        json.dumps(acceptance_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    acceptance_cv = score_acceptance(examples, oof_scores, thresholds)

    report = {
        "documents": len(documents),
        "folds": {str(index): [documents[item]["stem"] for item in fold] for index, fold in enumerate(folds)},
        "token_proposal_oof": token_oof,
        "span_examples": proposal_stats,
        "acceptance_oof": acceptance_cv,
        "thresholds": thresholds,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "format_version": 1,
        "source_documents": len(documents),
        "training_input": str(args.input_dir.resolve()),
        "training_gold": str(args.gold_dir.resolve()),
        "token_model": str(args.token_model.resolve()),
        "token_model_sha256": sha256_file(args.token_model),
        "acceptance_model": str(args.acceptance_model.resolve()),
        "acceptance_model_sha256": sha256_file(args.acceptance_model),
        "token_features": len(final_token_model.weights),
        "acceptance_features": len(weights),
        "report": str(args.report.resolve()),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "report": report}, ensure_ascii=False, indent=2))
    return 0


def load_documents(input_dir: Path, gold_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for gold_path in sorted(gold_dir.glob("*.json"), key=_natural_key):
        input_path = input_dir / f"{gold_path.stem}.txt"
        if not input_path.exists():
            continue
        text = input_path.read_text(encoding="utf-8-sig")
        raw = json.loads(gold_path.read_text(encoding="utf-8-sig"))
        concepts = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        valid = [item for item in concepts if _valid_gold(item, text)]
        tokens = tokenize(text)
        subsections = detect_subsections(text)
        output.append(
            {
                "stem": gold_path.stem,
                "text": text,
                "gold": valid,
                "tokens": tokens,
                "tags": gold_tags(tokens, valid),
                "subsections": [section_at(subsections, token.start) for token in tokens],
                "group": document_group(text, subsections),
            }
        )
    return output


def gold_tags(tokens: list[Token], concepts: list[dict[str, Any]]) -> list[str]:
    tags = ["O"] * len(tokens)
    ordered = sorted(
        concepts,
        key=lambda item: (item["position"][0], -(item["position"][1] - item["position"][0])),
    )
    for item in ordered:
        start, end = item["position"]
        indexes = [
            index
            for index, token in enumerate(tokens)
            if start <= token.start and token.end <= end and tags[index] == "O"
        ]
        if not indexes:
            continue
        concept_type = str(item["type"])
        tags[indexes[0]] = f"B:{concept_type}"
        for index in indexes[1:]:
            tags[index] = f"I:{concept_type}"
    return tags


def document_group(text: str, subsections: list[Any]) -> str:
    names = tuple(sorted({item.name for item in subsections if item.name != "document"}))
    lines = [line for line in text.splitlines() if line.strip()]
    cases = max(1, normalize_key(text).count("ma so ho so"))
    length_bucket = min(5, len(text) // 5000)
    line_bucket = min(5, len(lines) // 40)
    return json.dumps([names, min(cases, 5), length_bucket, line_bucket], ensure_ascii=True)


def grouped_folds(documents: list[dict[str, Any]], fold_count: int) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        groups[str(document["group"])].append(index)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for _, indexes in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(fold_count), key=lambda index: (len(folds[index]), index))
        folds[target].extend(indexes)
    return folds


def train_token_model(
    documents: Iterable[dict[str, Any]],
    labels: tuple[str, ...],
    epochs: int = 5,
) -> TokenSpanModel:
    sequences = [
        (document["tokens"], document["tags"], document["subsections"])
        for document in documents
    ]
    return AveragedPerceptronTrainer(labels).train(sequences, epochs=epochs)


def evaluate_token_oof(
    documents: list[dict[str, Any]],
    folds: list[list[int]],
    labels: tuple[str, ...],
    epochs: int = 3,
) -> dict[str, float | int]:
    tp = fp = fn = 0
    universe = set(range(len(documents)))
    for fold in folds:
        test = set(fold)
        train = [documents[index] for index in sorted(universe - test)]
        if not train or not test:
            continue
        model = train_token_model(train, labels, epochs=epochs)
        for index in sorted(test):
            document = documents[index]
            predicted = {(span.start, span.end, span.type) for span in model.propose(document["text"])}
            gold = {
                (item["position"][0], item["position"][1], str(item["type"]))
                for item in document["gold"]
            }
            tp += len(gold & predicted)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
    return _span_scores(tp, fp, fn)


def build_span_examples(
    documents: list[dict[str, Any]],
    token_model: TokenSpanModel,
    memory: AnnotationMemory,
    archived_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ner = MedicalNER()
    lattice = BoundaryVariantGenerator()
    examples: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    valid_archived_files = 0
    for document_id, document in enumerate(documents):
        text = str(document["text"])
        proposals = [*ner.propose(text), *memory.propose(text), *token_model.propose(text)]
        archived = _archived_spans(archived_dir, str(document["stem"]), text)
        if archived:
            valid_archived_files += 1
            proposals.extend(archived)
        proposals, _ = lattice.expand(text, proposals)
        gold = {
            (item["position"][0], item["position"][1], str(item["type"]))
            for item in document["gold"]
        }
        sections = detect_sections(text)
        subsections = detect_subsections(text)
        dedup: dict[tuple[int, int, str, str, str], SpanCandidate] = {}
        for span in proposals:
            key = (span.start, span.end, span.type, span.source, span.variant)
            current = dedup.get(key)
            if current is None or span.score > current.score:
                dedup[key] = span
        for span in dedup.values():
            source_counts[span.parent_source or span.source] += 1
            features = span_features(
                text,
                span,
                section_at(sections, span.start),
                section_at(subsections, span.start),
                None,
            )
            examples.append(
                {
                    "document_id": document_id,
                    "features": features,
                    "label": int((span.start, span.end, span.type) in gold),
                    "type": span.type,
                    "source": span.parent_source or span.source,
                    "variant": span.variant,
                }
            )
    return examples, {
        "examples": len(examples),
        "positive": sum(item["label"] for item in examples),
        "negative": sum(not item["label"] for item in examples),
        "by_source": dict(source_counts),
        "valid_archived_llm_files": valid_archived_files,
    }


def acceptance_oof(examples: list[dict[str, Any]], folds: list[list[int]]) -> list[float]:
    document_fold = {
        document_id: fold_index
        for fold_index, fold in enumerate(folds)
        for document_id in fold
    }
    scores = [0.0] * len(examples)
    for fold_index in range(len(folds)):
        train = [item for item in examples if document_fold.get(item["document_id"]) != fold_index]
        bias, weights = train_logistic(train, epochs=5)
        for index, item in enumerate(examples):
            if document_fold.get(item["document_id"]) != fold_index:
                continue
            logit = bias + sum(weights.get(feature, 0.0) for feature in item["features"])
            scores[index] = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
    return scores


def train_logistic(
    examples: list[dict[str, Any]],
    epochs: int = 14,
    max_positive_weight: float = 4.0,
) -> tuple[float, dict[str, float]]:
    positives = sum(item["label"] for item in examples)
    negatives = len(examples) - positives
    if not positives or not negatives:
        return 0.0, {}
    feature_support: Counter[str] = Counter(
        feature for item in examples for feature in set(item["features"])
    )
    allowed = {feature for feature, count in feature_support.items() if count >= 2}
    weights: dict[str, float] = defaultdict(float)
    bias = math.log((positives + 1.0) / (negatives + 1.0))
    positive_weight = min(max_positive_weight, negatives / max(1, positives))
    for epoch in range(epochs):
        learning_rate = 0.055 / (1.0 + 0.22 * epoch)
        ordered = examples if epoch % 2 == 0 else reversed(examples)
        for item in ordered:
            features = [feature for feature in item["features"] if feature in allowed]
            logit = bias + sum(weights[feature] for feature in features)
            probability = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
            sample_weight = positive_weight if item["label"] else 1.0
            gradient = (float(item["label"]) - probability) * sample_weight
            bias += learning_rate * gradient
            for feature in features:
                weights[feature] += learning_rate * (gradient - 0.0005 * weights[feature])
    return bias, {key: value for key, value in weights.items() if abs(value) >= 0.015}


def compact_training_examples(
    examples: list[dict[str, Any]],
    negative_ratio: int = 4,
) -> list[dict[str, Any]]:
    positives = [item for item in examples if item["label"]]
    hard_negatives = [
        item
        for item in examples
        if not item["label"] and (item["source"] == "llm" or item["variant"] != "original")
    ]
    hard_ids = {id(item) for item in hard_negatives}
    ordinary = [item for item in examples if not item["label"] and id(item) not in hard_ids]
    ordinary.sort(
        key=lambda item: hashlib.sha1(
            "|".join(item["features"]).encode("utf-8")
        ).hexdigest()
    )
    negative_limit = max(len(hard_negatives), negative_ratio * len(positives))
    selected_ordinary = ordinary[:max(0, negative_limit - len(hard_negatives))]
    output = [*positives, *hard_negatives, *selected_ordinary]
    return sorted(
        output,
        key=lambda item: (item["document_id"], -item["label"], item["type"], item["source"]),
    )


def tune_thresholds(examples: list[dict[str, Any]], scores: list[float]) -> dict[str, float]:
    output: dict[str, float] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(examples):
        groups[str(item["type"])].append(index)
        groups[f"{item['type']}|{item['source']}"] .append(index)
    for name, indexes in groups.items():
        if len(indexes) < 20 or sum(examples[index]["label"] for index in indexes) < 3:
            continue
        best = max(
            (value / 100.0 for value in range(30, 86, 2)),
            key=lambda threshold: _fbeta_for_indexes(examples, scores, indexes, threshold, beta=0.7),
        )
        output[name] = round(best, 2)
    return output


def score_acceptance(
    examples: list[dict[str, Any]],
    scores: list[float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    overall = _acceptance_counts(examples, scores, thresholds)
    by_type = {
        concept_type: _acceptance_counts(
            examples,
            scores,
            thresholds,
            indexes=[index for index, item in enumerate(examples) if item["type"] == concept_type],
        )
        for concept_type in ALLOWED_TYPES
    }
    return {**overall, "by_type": by_type}


def _acceptance_counts(
    examples: list[dict[str, Any]],
    scores: list[float],
    thresholds: dict[str, float],
    indexes: list[int] | None = None,
) -> dict[str, float | int]:
    tp = fp = fn = 0
    for index in indexes if indexes is not None else range(len(examples)):
        item = examples[index]
        threshold = thresholds.get(
            f"{item['type']}|{item['source']}", thresholds.get(str(item["type"]), 0.5)
        )
        predicted = scores[index] >= threshold
        tp += int(predicted and item["label"])
        fp += int(predicted and not item["label"])
        fn += int(not predicted and item["label"])
    return _span_scores(tp, fp, fn)


def _fbeta_for_indexes(
    examples: list[dict[str, Any]],
    scores: list[float],
    indexes: list[int],
    threshold: float,
    beta: float,
) -> float:
    tp = sum(scores[index] >= threshold and examples[index]["label"] for index in indexes)
    fp = sum(scores[index] >= threshold and not examples[index]["label"] for index in indexes)
    fn = sum(scores[index] < threshold and examples[index]["label"] for index in indexes)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / max(1e-9, beta2 * precision + recall)


def _archived_spans(directory: Path | None, stem: str, text: str) -> list[SpanCandidate]:
    if directory is None:
        return []
    path = directory / f"{stem}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    concepts = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    valid = [item for item in concepts if _valid_gold(item, text)]
    if not concepts or len(valid) / len(concepts) < 0.8:
        return []
    return [
        SpanCandidate(
            item["position"][0],
            item["position"][1],
            str(item["text"]),
            str(item["type"]),
            0.72,
            "llm",
        )
        for item in valid
    ]


def _valid_gold(item: dict[str, Any], text: str) -> bool:
    position = item.get("position")
    concept_type = str(item.get("type") or "")
    return bool(
        concept_type in ALLOWED_TYPES
        and isinstance(position, list)
        and len(position) == 2
        and all(isinstance(value, int) for value in position)
        and 0 <= position[0] < position[1] <= len(text)
        and text[position[0]:position[1]] == item.get("text")
    )


def _span_scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (sys.maxsize, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
