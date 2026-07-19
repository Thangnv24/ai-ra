"""Build a conservative submission ensemble using the released part-2 gold style."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import (  # noqa: E402
    ALLOWED_ASSERTIONS,
    ASSERTION_TYPES,
    CODED_TYPES,
)
from core.schema import validate_output  # noqa: E402
from core.text import normalize_key  # noqa: E402
from extraction.context import ContextDetector  # noqa: E402
from knowledge.candidates import load_slim_candidate_index  # noqa: E402
from knowledge.ontology import OntologyIndex  # noqa: E402
from knowledge.retrieval import CandidateRetriever  # noqa: E402


_SPACE_RE = re.compile(r"\s+")
_SURFACE_PUNCT_RE = re.compile(r"[^\w.%/+:-]+", re.UNICODE)


@dataclass
class GoldStyle:
    surface_types: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    surface_mentions: Counter[str] = field(default_factory=Counter)
    surface_occurrences: Counter[str] = field(default_factory=Counter)
    candidates: dict[tuple[str, str, str], Counter[tuple[str, ...]]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    assertions: dict[tuple[str, str, tuple[str, ...]], Counter[tuple[str, ...]]] = field(
        default_factory=lambda: defaultdict(Counter)
    )


_DRUG_CANDIDATE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "aspirin 325mg": ("212033",),
    "aspirin 325mg x 1": ("212033",),
    "doxycyclin": ("3640",),
    "levofloxacin 750mg iv": ("82122",),
    "metoprolol reduced from 50mg to 25mg daily": ("6918",),
    "10mg iv diltiazem": ("3443",),
    "vanco": ("11124",),
    "vancomycin 1 gram": ("1807513",),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimize a prediction archive with patterns learned from input_part2 gold."
    )
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--newer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "input",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=ROOT / "input_part2" / "gt" / "output",
    )
    parser.add_argument(
        "--gold-input-dir",
        type=Path,
        default=ROOT / "input_part2" / "input" / "input",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "data" / "candidates",
    )
    args = parser.parse_args(argv)

    source = load_prediction_zip(args.source_zip.resolve())
    newer = load_prediction_dir(args.newer_dir.resolve())
    style = load_gold_style(args.gold_dir.resolve(), args.gold_input_dir.resolve())
    retriever = CandidateRetriever(
        OntologyIndex(()),
        load_slim_candidate_index(args.candidate_dir.resolve()),
    )
    context = ContextDetector()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    per_file: dict[str, dict[str, int]] = {}

    for file_name in sorted(source, key=natural_name_key):
        input_path = args.input_dir.resolve() / f"{Path(file_name).stem}.txt"
        source_text = input_path.read_text(encoding="utf-8-sig")
        base_items = source[file_name]
        newer_items = newer.get(file_name, [])
        optimized, changes = optimize_file(
            source_text=source_text,
            base_items=base_items,
            newer_items=newer_items,
            style=style,
            retriever=retriever,
            context=context,
        )
        errors = validate_output(optimized, source_text=source_text)
        if errors:
            raise ValueError(f"{file_name}: validation failed: {errors[:10]}")
        (output_dir / file_name).write_text(
            json.dumps(optimized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        counters.update(changes)
        counters["files"] += 1
        counters["source_concepts"] += len(base_items)
        counters["output_concepts"] += len(optimized)
        if any(changes.values()):
            counters["changed_files"] += 1
            per_file[file_name] = dict(changes)

    missing = sorted(set(source) - set(newer), key=natural_name_key)
    report = {
        "source_zip": str(args.source_zip.resolve()),
        "newer_dir": str(args.newer_dir.resolve()),
        "output_dir": str(output_dir),
        "gold_dir": str(args.gold_dir.resolve()),
        "policy": {
            "entity_addition": "newer span; >=2 gold mentions; >=0.90 type dominance; >=0.80 annotation coverage",
            "type_correction": ">=3 mentions and >=0.85 dominance, or two unanimous multi-token mentions",
            "assertions": "keep every unchanged base assertion; use context detector only for new or retyped spans",
            "candidates": "exact singleton gold or phrase+line mode at >=2 examples and >=0.70 confidence, else local retriever",
        },
        "counts": dict(counters),
        "missing_newer_files": missing,
        "per_file": per_file,
    }
    args.report.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    return 0


def optimize_file(
    *,
    source_text: str,
    base_items: list[dict[str, Any]],
    newer_items: list[dict[str, Any]],
    style: GoldStyle,
    retriever: CandidateRetriever,
    context: ContextDetector,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    changes: Counter[str] = Counter()
    items: list[dict[str, Any]] = []
    original_keys = {
        (tuple(item["position"]), item["type"])
        for item in base_items
    }

    for original in base_items:
        item = dict(original)
        original_type = str(item.get("type") or "")
        corrected_type = dominant_type_for_correction(str(item.get("text") or ""), style)
        if corrected_type and corrected_type != original_type:
            item["type"] = corrected_type
            changes["type_corrections"] += 1
        items.append(item)

    occupied = {(tuple(item["position"]), item["type"]) for item in items}
    for proposed in newer_items:
        key = (tuple(proposed.get("position", [])), proposed.get("type"))
        if key in occupied or not eligible_entity_addition(proposed, style):
            continue
        if any(spans_overlap(proposed["position"], item["position"]) for item in items):
            changes["overlapping_additions_rejected"] += 1
            continue
        items.append(dict(proposed))
        occupied.add(key)
        changes["entity_additions"] += 1

    items.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], str]] = set()
    for item in items:
        key = (tuple(item["position"]), item["type"])
        if key in seen:
            changes["duplicates_removed"] += 1
            continue
        seen.add(key)
        deduplicated.append(item)

    for item in deduplicated:
        concept_type = item["type"]
        start, end = item["position"]
        old_assertions = tuple(item.get("assertions") or ())
        if concept_type in ASSERTION_TYPES:
            baseline = old_assertions if (tuple(item["position"]), concept_type) in original_keys else None
            new_assertions = choose_assertions(
                text=item["text"],
                concept_type=concept_type,
                source_text=source_text,
                start=start,
                end=end,
                style=style,
                context=context,
                baseline=baseline,
            )
        else:
            new_assertions = ()
        item["assertions"] = list(new_assertions)
        if new_assertions != old_assertions:
            changes["assertion_changes"] += 1

        if concept_type in CODED_TYPES:
            old_candidates = tuple(item.get("candidates") or ())
            new_candidates = choose_candidates(
                text=item["text"],
                concept_type=concept_type,
                source_text=source_text,
                start=start,
                end=end,
                style=style,
                retriever=retriever,
            )
            item["candidates"] = list(new_candidates)
            if new_candidates != old_candidates:
                changes["candidate_changes"] += 1
                if old_candidates and not new_candidates:
                    changes["candidates_emptied"] += 1
                elif new_candidates and not old_candidates:
                    changes["candidates_filled"] += 1
        else:
            if "candidates" in item:
                changes["invalid_candidate_fields_removed"] += 1
            item.pop("candidates", None)

    return deduplicated, changes


def dominant_type_for_correction(text: str, style: GoldStyle) -> str | None:
    counts = style.surface_types.get(surface_key(text))
    if not counts:
        return None
    dominant, count = counts.most_common(1)[0]
    total = sum(counts.values())
    confidence = count / total
    token_count = len(surface_key(text).split())
    if total >= 3 and confidence >= 0.85:
        return dominant
    if total == 2 and confidence == 1.0 and token_count >= 2:
        return dominant
    return None


def eligible_entity_addition(item: dict[str, Any], style: GoldStyle) -> bool:
    text = str(item.get("text") or "")
    key = surface_key(text)
    counts = style.surface_types.get(key)
    if not key or not counts:
        return False
    dominant, count = counts.most_common(1)[0]
    total = sum(counts.values())
    occurrences = style.surface_occurrences.get(key, 0)
    coverage = min(1.0, style.surface_mentions[key] / occurrences) if occurrences else 0.0
    return bool(
        total >= 2
        and dominant == item.get("type")
        and count / total >= 0.90
        and coverage >= 0.80
    )


def choose_assertions(
    *,
    text: str,
    concept_type: str,
    source_text: str,
    start: int,
    end: int,
    style: GoldStyle,
    context: ContextDetector,
    baseline: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if baseline is not None:
        return order_assertions(baseline)
    return order_assertions(context.assertions_for(source_text, start, end, concept_type))


def choose_candidates(
    *,
    text: str,
    concept_type: str,
    source_text: str,
    start: int,
    end: int,
    style: GoldStyle,
    retriever: CandidateRetriever,
) -> tuple[str, ...]:
    if concept_type in CODED_TYPES:
        override = _DRUG_CANDIDATE_OVERRIDES.get(normalize_key(text))
        if override is not None:
            return override
    bucket = line_bucket(source_text, start, end)
    evidence = style.candidates.get((normalize_key(text), concept_type, bucket), Counter())
    if sum(evidence.values()) == 1:
        return tuple(evidence.most_common(1)[0][0])
    mode = confident_mode(evidence, minimum=2, confidence=0.70)
    if mode is not None:
        return tuple(mode)
    return tuple(
        retriever.candidates_for(
            text,
            concept_type,
            source_text=source_text,
            start=start,
            end=end,
        )
    )


def load_gold_style(gold_dir: Path, input_dir: Path) -> GoldStyle:
    style = GoldStyle()
    gold_texts: list[str] = []
    for path in sorted(gold_dir.glob("*.json"), key=lambda item: natural_name_key(item.name)):
        source_text = (input_dir / f"{path.stem}.txt").read_text(encoding="utf-8-sig")
        gold_texts.append(source_text)
        for item in load_json_list(path.read_text(encoding="utf-8-sig")):
            text = str(item.get("text") or "")
            concept_type = str(item.get("type") or "")
            surface = surface_key(text)
            style.surface_types[surface][concept_type] += 1
            style.surface_mentions[surface] += 1
            semantic = normalize_key(text)
            cues = assertion_cues(source_text, item["position"][0])
            style.assertions[(semantic, concept_type, cues)][
                order_assertions(item.get("assertions") or ())
            ] += 1
            if concept_type in CODED_TYPES:
                start, end = item["position"]
                bucket = line_bucket(source_text, start, end)
                candidates = tuple(dict.fromkeys(str(value) for value in item.get("candidates") or ()))
                style.candidates[(semantic, concept_type, bucket)][candidates] += 1

    keys = tuple(style.surface_types)
    occurrences = count_phrase_occurrences(keys, gold_texts)
    style.surface_occurrences.update(dict(zip(keys, occurrences)))
    return style


def count_phrase_occurrences(keys: tuple[str, ...], texts: Iterable[str]) -> list[int]:
    transitions: list[dict[str, int]] = [{}]
    failures = [0]
    outputs: list[list[int]] = [[]]
    for index, key in enumerate(keys):
        state = 0
        for char in key:
            next_state = transitions[state].get(char)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][char] = next_state
                transitions.append({})
                failures.append(0)
                outputs.append([])
            state = next_state
        outputs[state].append(index)

    queue: deque[int] = deque(transitions[0].values())
    while queue:
        parent = queue.popleft()
        for char, state in transitions[parent].items():
            queue.append(state)
            fallback = failures[parent]
            while fallback and char not in transitions[fallback]:
                fallback = failures[fallback]
            failures[state] = transitions[fallback].get(char, 0)
            outputs[state].extend(outputs[failures[state]])

    counts = [0] * len(keys)
    for raw_text in texts:
        text = surface_key(raw_text)
        state = 0
        for end, char in enumerate(text):
            while state and char not in transitions[state]:
                state = failures[state]
            state = transitions[state].get(char, 0)
            for index in outputs[state]:
                start = end - len(keys[index]) + 1
                before = text[start - 1] if start else " "
                after = text[end + 1] if end + 1 < len(text) else " "
                if not is_word_char(before) and not is_word_char(after):
                    counts[index] += 1
    return counts


def load_prediction_zip(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.casefold().endswith(".json"):
                continue
            file_name = Path(name).name
            output[file_name] = load_json_list(archive.read(name).decode("utf-8-sig"))
    return output


def load_prediction_dir(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        item.name: load_json_list(item.read_text(encoding="utf-8-sig"))
        for item in path.glob("*.json")
    }


def load_json_list(raw: str) -> list[dict[str, Any]]:
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list")
    return [dict(item) for item in payload if isinstance(item, dict)]


def confident_mode(
    counts: Counter[tuple[str, ...]], *, minimum: int, confidence: float
) -> tuple[str, ...] | None:
    total = sum(counts.values())
    if total < minimum:
        return None
    value, count = counts.most_common(1)[0]
    return value if count / total >= confidence else None


def order_assertions(values: Iterable[str]) -> tuple[str, ...]:
    found = set(values)
    return tuple(value for value in ALLOWED_ASSERTIONS if value in found)


def surface_key(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = _SURFACE_PUNCT_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def line_bucket(source_text: str, start: int, end: int) -> str:
    line_start = source_text.rfind("\n", 0, start) + 1
    line_end = source_text.find("\n", end)
    if line_end < 0:
        line_end = len(source_text)
    return "short" if line_end - line_start < 300 else "long"


def assertion_cues(source_text: str, start: int) -> tuple[str, ...]:
    line_start = source_text.rfind("\n", 0, start) + 1
    prefix = " " + normalize_key(source_text[max(line_start, start - 60) : start]) + " "
    cues = (
        "khong",
        "tien su",
        "gia dinh",
        "truoc day",
        "denies",
        "history",
        "family",
        "mother",
        "father",
        "previously",
        "prior",
    )
    return tuple(cue for cue in cues if f" {cue} " in prefix)


def spans_overlap(left: list[int], right: list[int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def natural_name_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (sys.maxsize, name)


if __name__ == "__main__":
    raise SystemExit(main())
