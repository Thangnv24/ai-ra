"""Build reusable contextual annotation evidence from paired text and gold data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_TYPES
from core.text import normalize_key
from extraction.annotation_memory import NormalizedKeyMatcher, context_cues, normalized_projection, section_at
from extraction.sectioning import detect_sections, detect_subsections


STOP_KEYS = {
    "benh nhan",
    "chan doan",
    "chan doan hinh anh",
    "danh gia",
    "danh gia ban dau",
    "danh gia lam sang",
    "dau hieu",
    "dieu tri",
    "tinh",
    "trieu chung",
    "xet nghiem",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build contextual annotation memory from reviewed corpora.")
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("INPUT_DIR", "GOLD_DIR"),
        help="Paired directories. May be repeated as more reviewed datasets become available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "external" / "annotation_memory.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "external" / "annotation_memory_manifest.json",
    )
    args = parser.parse_args(argv)
    datasets = args.dataset or [
        [str(ROOT / "input_part2" / "input" / "input"), str(ROOT / "input_part2" / "gt" / "output")]
    ]
    pairs = [(Path(input_dir).resolve(), Path(gold_dir).resolve()) for input_dir, gold_dir in datasets]
    rows, statistics = build_memory(pairs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "format_version": 2,
        "datasets": [
            {"input_dir": str(input_dir), "gold_dir": str(gold_dir)} for input_dir, gold_dir in pairs
        ],
        "selection": {
            "runtime_min_positive_count": 3,
            "runtime_min_support_documents": 2,
            "runtime_min_type_purity": 0.85,
            "runtime_min_multi_token_annotation_rate": 0.62,
            "runtime_min_single_token_positive_count": 8,
            "runtime_min_single_token_annotation_rate": 0.8,
        },
        **statistics,
        "artifact": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
    }
    with args.manifest.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_memory(
    datasets: list[tuple[Path, Path]],
    *,
    include_stems: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    assertions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    candidates: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    candidate_sets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    positive_spans: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    positive_documents: dict[str, set[int]] = defaultdict(set)
    cue_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    negative_cue_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    boundary_profiles: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    provenance: dict[str, set[str]] = defaultdict(set)

    for dataset_index, (input_dir, gold_dir) in enumerate(datasets, start=1):
        for gold_path in sorted(gold_dir.glob("*.json"), key=natural_key):
            if include_stems is not None and gold_path.stem not in include_stems:
                continue
            input_path = input_dir / f"{gold_path.stem}.txt"
            if not input_path.exists():
                continue
            text = input_path.read_text(encoding="utf-8-sig")
            raw = json.loads(gold_path.read_text(encoding="utf-8-sig"))
            document_id = len(documents)
            sections = detect_sections(text)
            subsections = detect_subsections(text)
            concepts = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
            for item in concepts:
                concept_type = str(item.get("type") or "")
                position = item.get("position")
                term = str(item.get("text") or "")
                key = normalize_key(term)
                if concept_type not in ALLOWED_TYPES or not eligible_key(key):
                    continue
                if not valid_position(position, text, term):
                    continue
                start, end = int(position[0]), int(position[1])
                positive_spans[(document_id, start, end)].add(key)
                positive_documents[key].add(document_id)
                type_counts[key][concept_type] += 1
                surfaces[key][term] += 1
                section_name = section_at(sections, start)
                subsection_name = section_at(subsections, start)
                left, right = context_cues(text, start, end)
                cue_counts[(key, "left")].update(left)
                cue_counts[(key, "right")].update(right)
                assertion_key = "|".join(sorted(str(value) for value in item.get("assertions") or ()))
                evidence_key = (key, concept_type)
                assertions[evidence_key][assertion_key] += 1
                candidate_values = tuple(sorted(str(value) for value in item.get("candidates") or ()))
                candidates[evidence_key].update(candidate_values)
                candidate_sets[evidence_key]["|".join(candidate_values)] += 1
                boundary_profiles[evidence_key].update(_boundary_features(term))
                provenance[key].add(f"dataset_{dataset_index}")
                _append_example(
                    examples[evidence_key],
                    text,
                    start,
                    end,
                    term,
                    section_name,
                    subsection_name,
                )
            documents.append({"text": text, "sections": sections, "subsections": subsections})

    observed_count: Counter[str] = Counter()
    observed_documents: dict[str, set[int]] = defaultdict(set)
    positive_occurrences: Counter[str] = Counter()
    section_observed: dict[str, Counter[str]] = defaultdict(Counter)
    section_positive: dict[str, Counter[str]] = defaultdict(Counter)
    subsection_observed: dict[str, Counter[str]] = defaultdict(Counter)
    subsection_positive: dict[str, Counter[str]] = defaultdict(Counter)
    keys = tuple(type_counts)
    matcher = NormalizedKeyMatcher(keys)
    for document_id, document in enumerate(documents):
        text = str(document["text"])
        projection = normalized_projection(text)
        sections = document["sections"]
        subsections = document["subsections"]
        for key, start, end in matcher.find(projection):
            observed_documents[key].add(document_id)
            observed_count[key] += 1
            section_name = section_at(sections, start)
            subsection_name = section_at(subsections, start)
            section_observed[key][section_name] += 1
            subsection_observed[key][subsection_name] += 1
            if key in positive_spans.get((document_id, start, end), set()):
                positive_occurrences[key] += 1
                section_positive[key][section_name] += 1
                subsection_positive[key][subsection_name] += 1
            else:
                left, right = context_cues(text, start, end)
                negative_cue_counts[(key, "left")].update(left)
                negative_cue_counts[(key, "right")].update(right)

    rows: list[dict[str, Any]] = []
    for key in keys:
        counts = type_counts[key]
        concept_type, dominant_count = counts.most_common(1)[0]
        evidence_key = (key, concept_type)
        total_typed = sum(counts.values())
        positives = positive_occurrences[key] or len(positive_documents[key])
        observed = max(positives, observed_count[key])
        rows.append(
            {
                "key": key,
                "surfaces": [value for value, _ in surfaces[key].most_common(5)],
                "type": concept_type,
                "type_counts": dict(counts),
                "positive_count": positives,
                "observed_count": observed,
                "negative_count": max(0, observed - positives),
                "support_documents": len(positive_documents[key]),
                "observed_documents": len(observed_documents[key]),
                "type_purity": round(dominant_count / max(1, total_typed), 6),
                "annotation_rate": round(positives / max(1, observed), 6),
                "sections": {
                    name: {
                        "positive": section_positive[key][name],
                        "observed": count,
                    }
                    for name, count in sorted(section_observed[key].items())
                },
                "left_cues": [value for value, _ in cue_counts[(key, "left")].most_common(8)],
                "right_cues": [value for value, _ in cue_counts[(key, "right")].most_common(8)],
                "negative_left_cues": [
                    value for value, _ in negative_cue_counts[(key, "left")].most_common(8)
                ],
                "negative_right_cues": [
                    value for value, _ in negative_cue_counts[(key, "right")].most_common(8)
                ],
                "subsections": {
                    name: {
                        "positive": subsection_positive[key][name],
                        "observed": count,
                    }
                    for name, count in sorted(subsection_observed[key].items())
                },
                "boundary_profile": dict(boundary_profiles[evidence_key]),
                "examples": examples[evidence_key][:3],
                "assertions": dict(assertions[evidence_key]),
                "candidates": dict(candidates[evidence_key].most_common()),
                "candidate_sets": dict(candidate_sets[evidence_key]),
                "provenance": {
                    "kind": "reviewed_gold",
                    "datasets": sorted(provenance[key]),
                },
            }
        )
    rows.sort(key=lambda item: (-int(item["positive_count"]), str(item["key"])))
    return rows, {
        "source_documents": len(documents),
        "row_count": len(rows),
        "positive_occurrences": sum(int(row["positive_count"]) for row in rows),
        "hard_negative_occurrences": sum(int(row["negative_count"]) for row in rows),
        "rows_with_hard_negative_context": sum(
            bool(row["negative_left_cues"] or row["negative_right_cues"]) for row in rows
        ),
        "rows_with_examples": sum(bool(row["examples"]) for row in rows),
        "by_type": dict(Counter(str(row["type"]) for row in rows)),
    }


def _append_example(
    output: list[dict[str, str]],
    text: str,
    start: int,
    end: int,
    quote: str,
    section: str,
    subsection: str,
) -> None:
    if len(output) >= 3:
        return
    left = text[max(0, start - 80) : start].replace("\n", " ").strip()
    right = text[end : min(len(text), end + 80)].replace("\n", " ").strip()
    item = {
        "quote": quote,
        "left": left[-80:],
        "right": right[:80],
        "section": section,
        "subsection": subsection,
    }
    identity = (item["quote"], item["left"], item["right"])
    if all((value["quote"], value["left"], value["right"]) != identity for value in output):
        output.append(item)


def _boundary_features(value: str) -> Counter[str]:
    key = normalize_key(value)
    features: Counter[str] = Counter()
    features[f"tokens:{len(key.split())}"] += 1
    if key.startswith(("khong ", "chua ", "phu nhan ")):
        features["leading_negation"] += 1
    if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b", key):
        features["has_strength_or_unit"] += 1
    if "(" in value and ")" in value:
        features["has_parenthetical"] += 1
    if re.search(r"\b(?:po|iv|im|sc|bid|tid|qid|qhs|prn)\b", key):
        features["has_sig"] += 1
    return features


def eligible_key(key: str) -> bool:
    if len(key) < 3 or key in STOP_KEYS:
        return False
    if len(key.split()) == 1 and len(key) < 4:
        return False
    return True


def valid_position(position: Any, text: str, term: str) -> bool:
    return bool(
        isinstance(position, list)
        and len(position) == 2
        and all(isinstance(value, int) for value in position)
        and 0 <= position[0] < position[1] <= len(text)
        and text[position[0] : position[1]] == term
    )


def natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (sys.maxsize, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
