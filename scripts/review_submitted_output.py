"""Format and audit a submitted output folder against the official inputs.

This script does not run inference. It uses an existing prediction folder as the
baseline, writes a clean submission-shaped copy, and creates per-file review
notes so we can inspect the weak files one by one.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.config import (  # noqa: E402
    ALLOWED_ASSERTIONS,
    ALLOWED_TYPES,
    ASSERTION_TYPES,
    CODED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
)
from core.schema import validate_output  # noqa: E402


DRUG_HINT_RE = re.compile(
    r"\b[A-Z][A-Za-z][A-Za-z0-9/-]*(?:\s+[A-Za-z0-9./:%-]+){0,8}"
    r"(?:\s+\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b|\s+(?:po|iv|oral|tablet|capsule|solution|syrup|cream|bid|qid|qhs|qam|prn)\b)",
    re.IGNORECASE,
)
LAB_VALUE_RE = re.compile(r"\b[A-Z][A-Z0-9%/-]{1,12}\s*[:=]\s*-?\d+(?:[.,]\d+)?")
SUSPICIOUS_DIAGNOSIS_RE = re.compile(
    r"\b(hình ảnh|x-quang|x quang|ct|mri|siêu âm|không có gì đáng chú ý|khác)\b",
    re.IGNORECASE,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem) if path.stem.isdigit() else 10**9, path.name)


def load_candidate_names(candidate_dir: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for filename in ("icd10_candidates.jsonl", "rxnorm_candidates.jsonl"):
        path = candidate_dir / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                code = str(row.get("code", ""))
                name = str(row.get("name") or row.get("name_vi") or row.get("name_en") or code)
                if code:
                    names.setdefault(code, name)
    return names


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    concept_type = raw.get("type")
    assertions = []
    for assertion in raw.get("assertions") or []:
        if assertion in ALLOWED_ASSERTIONS and assertion not in assertions:
            assertions.append(assertion)

    position = raw.get("position") or [0, 0]
    if isinstance(position, tuple):
        position = list(position)
    if not isinstance(position, list) or len(position) != 2:
        position = [0, 0]

    item: dict[str, Any] = {
        "text": str(raw.get("text", "")),
        "type": str(concept_type),
    }

    if concept_type in CODED_TYPES:
        candidates: list[str] = []
        for candidate in raw.get("candidates") or []:
            code = str(candidate)
            if code and code not in candidates:
                candidates.append(code)
        item["candidates"] = candidates

    item["assertions"] = assertions if concept_type in ASSERTION_TYPES else []
    item["position"] = [int(position[0]), int(position[1])]
    return item


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    items = [normalize_item(item) for item in payload if isinstance(item, dict)]
    items.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"], item["text"]))
    return items


def package_zip(formatted_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(formatted_dir.glob("*.json"), key=sort_key)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=f"output/{path.name}")


def span_context(source_text: str, start: int, end: int, radius: int = 70) -> str:
    left = max(0, start - radius)
    right = min(len(source_text), end + radius)
    prefix = "..." if left else ""
    suffix = "..." if right < len(source_text) else ""
    return prefix + source_text[left:start] + "[[" + source_text[start:end] + "]]" + source_text[end:right] + suffix


def md_escape(value: Any, max_len: int = 220) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def candidate_label(candidates: list[str], candidate_names: dict[str, str]) -> str:
    labels = []
    for code in candidates[:5]:
        name = candidate_names.get(code, "?")
        labels.append(f"{code}:{name}")
    return "; ".join(labels)


def overlap_count(items: list[dict[str, Any]]) -> int:
    count = 0
    sorted_items = sorted(items, key=lambda item: (item["position"][0], item["position"][1]))
    for i, left in enumerate(sorted_items):
        ls, le = left["position"]
        for right in sorted_items[i + 1 :]:
            rs, re_ = right["position"]
            if rs >= le:
                break
            if left["type"] == right["type"] and (ls, le) != (rs, re_):
                count += 1
    return count


def audit_file(
    fid: int,
    source_text: str,
    items: list[dict[str, Any]],
    candidate_names: dict[str, str],
) -> dict[str, Any]:
    errors = validate_output(items, source_text=source_text)
    counts = Counter(str(item.get("type")) for item in items)
    assertion_counts = Counter(
        assertion for item in items for assertion in item.get("assertions", [])
    )
    bad_candidates = []
    wide_candidates = 0
    empty_coded_candidates = 0
    suspicious_diagnoses = []

    for item in items:
        concept_type = item.get("type")
        candidates = item.get("candidates") or []
        if concept_type in CODED_TYPES:
            if not candidates:
                empty_coded_candidates += 1
            if len(candidates) > 2:
                wide_candidates += 1
            for code in candidates:
                if code not in candidate_names:
                    bad_candidates.append(code)
        if concept_type == TYPE_DIAGNOSIS:
            text = str(item.get("text", ""))
            if len(text) > 80 or SUSPICIOUS_DIAGNOSIS_RE.search(text):
                suspicious_diagnoses.append(text)

    per_1k = round(len(items) * 1000 / max(1, len(source_text)), 1)
    drug_hints = len(DRUG_HINT_RE.findall(source_text))
    lab_value_hints = len(LAB_VALUE_RE.findall(source_text))
    overlaps = overlap_count(items)

    flags: list[str] = []
    if errors:
        flags.append("schema_or_position_error")
    if len(items) == 0:
        flags.append("empty_output")
    if per_1k < 3:
        flags.append("very_sparse")
    elif per_1k < 5 and len(source_text) > 800:
        flags.append("under_extract")
    if drug_hints and counts.get(TYPE_DRUG, 0) == 0:
        flags.append("drug_hints_but_no_drug")
    if lab_value_hints and counts.get(TYPE_TEST_RESULT, 0) == 0:
        flags.append("lab_values_but_no_results")
    if counts.get(TYPE_TEST_NAME, 0) and counts.get(TYPE_TEST_RESULT, 0) == 0:
        flags.append("lab_names_without_results")
    if abs(counts.get(TYPE_TEST_NAME, 0) - counts.get(TYPE_TEST_RESULT, 0)) >= 4:
        flags.append("lab_name_result_imbalance")
    if empty_coded_candidates:
        flags.append("coded_without_candidates")
    if bad_candidates:
        flags.append("unknown_candidate_code")
    if wide_candidates:
        flags.append("wide_candidates")
    if overlaps:
        flags.append("overlapping_same_type")
    if suspicious_diagnoses:
        flags.append("suspicious_diagnosis_span")

    risk_score = (
        len(errors) * 4
        + int("very_sparse" in flags) * 5
        + int("under_extract" in flags) * 3
        + drug_hints * int("drug_hints_but_no_drug" in flags)
        + lab_value_hints * int("lab_values_but_no_results" in flags)
        + wide_candidates * 2
        + overlaps
        + len(suspicious_diagnoses) * 2
    )

    return {
        "file": fid,
        "chars": len(source_text),
        "items": len(items),
        "per_1k": per_1k,
        "diagnosis": counts.get(TYPE_DIAGNOSIS, 0),
        "symptom": counts.get("TRIỆU_CHỨNG", 0),
        "drug": counts.get(TYPE_DRUG, 0),
        "test_name": counts.get(TYPE_TEST_NAME, 0),
        "test_result": counts.get(TYPE_TEST_RESULT, 0),
        "isNegated": assertion_counts.get("isNegated", 0),
        "isFamily": assertion_counts.get("isFamily", 0),
        "isHistorical": assertion_counts.get("isHistorical", 0),
        "drug_hints": drug_hints,
        "lab_value_hints": lab_value_hints,
        "wide_candidates": wide_candidates,
        "overlaps": overlaps,
        "bad_candidates": len(set(bad_candidates)),
        "suspicious_diagnoses": len(suspicious_diagnoses),
        "errors": len(errors),
        "flags": ",".join(flags) if flags else "ok",
        "risk_score": risk_score,
        "error_examples": errors[:5],
        "suspicious_examples": suspicious_diagnoses[:5],
    }


def write_review_note(
    path: Path,
    fid: int,
    source_text: str,
    items: list[dict[str, Any]],
    summary: dict[str, Any],
    candidate_names: dict[str, str],
) -> None:
    lines = [
        f"# File {fid}",
        "",
        "## Audit",
        "",
        f"- chars: {summary['chars']}",
        f"- items: {summary['items']} ({summary['per_1k']} / 1k chars)",
        f"- counts: diagnosis={summary['diagnosis']}, symptom={summary['symptom']}, drug={summary['drug']}, test_name={summary['test_name']}, test_result={summary['test_result']}",
        f"- assertions: isNegated={summary['isNegated']}, isFamily={summary['isFamily']}, isHistorical={summary['isHistorical']}",
        f"- flags: {summary['flags']}",
        f"- risk_score: {summary['risk_score']}",
        "",
    ]
    if summary["error_examples"]:
        lines.extend(["## Schema / Position Errors", ""])
        lines.extend(f"- {err}" for err in summary["error_examples"])
        lines.append("")
    if summary["suspicious_examples"]:
        lines.extend(["## Suspicious Diagnosis Spans", ""])
        lines.extend(f"- {text}" for text in summary["suspicious_examples"])
        lines.append("")

    lines.extend(
        [
            "## Input",
            "",
            "```text",
            source_text,
            "```",
            "",
            "## Current Output",
            "",
            "|#|span|type|assertions|candidates|text|context|",
            "|---:|---:|---|---|---|---|---|",
        ]
    )
    for index, item in enumerate(items, 1):
        start, end = item["position"]
        candidates = item.get("candidates") or []
        lines.append(
            "|"
            + "|".join(
                [
                    str(index),
                    f"{start}-{end}",
                    md_escape(item.get("type", "")),
                    md_escape(",".join(item.get("assertions") or []), 80),
                    md_escape(candidate_label(candidates, candidate_names), 180),
                    md_escape(item.get("text", ""), 120),
                    md_escape(span_context(source_text, start, end), 220),
                ]
            )
            + "|"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    review_dir: Path,
    rows: list[dict[str, Any]],
    source_output: Path,
    formatted_dir: Path,
    zip_path: Path | None,
) -> None:
    rows_by_risk = sorted(rows, key=lambda row: (-int(row["risk_score"]), int(row["file"])))
    flags = Counter()
    for row in rows:
        for flag in str(row["flags"]).split(","):
            if flag and flag != "ok":
                flags[flag] += 1

    lines = [
        "# Submitted Output Review",
        "",
        f"- source_output: `{source_output}`",
        f"- formatted_output: `{formatted_dir}`",
        f"- packaged_zip: `{zip_path}`" if zip_path else "- packaged_zip: not built",
        f"- files: {len(rows)}",
        f"- total_items: {sum(int(row['items']) for row in rows)}",
        "",
        "## Flag Counts",
        "",
    ]
    if flags:
        lines.extend(f"- {flag}: {count}" for flag, count in flags.most_common())
    else:
        lines.append("- ok: no heuristic flags")
    lines.extend(
        [
            "",
            "## Highest Risk Files",
            "",
            "|file|risk|items|/1k|diag|sym|drug|labN|labR|drug_hint|lab_hint|flags|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows_by_risk[:35]:
        lines.append(
            f"|{row['file']}|{row['risk_score']}|{row['items']}|{row['per_1k']}|"
            f"{row['diagnosis']}|{row['symptom']}|{row['drug']}|{row['test_name']}|{row['test_result']}|"
            f"{row['drug_hints']}|{row['lab_value_hints']}|{row['flags']}|"
        )
    lines.extend(
        [
            "",
            "## Full Review Queue",
            "",
            "|rank|file|risk|flags|review_note|",
            "|---:|---:|---:|---|---|",
        ]
    )
    for rank, row in enumerate(rows_by_risk, 1):
        fid = int(row["file"])
        lines.append(
            f"|{rank}|{fid}|{row['risk_score']}|{row['flags']}|[{fid:03d}.md]({fid:03d}.md)|"
        )
    lines.extend(
        [
            "",
            "## Major Error Interpretation",
            "",
            "- JSON/position format is not the main bottleneck when `errors=0`; low score then comes from semantic mismatch with the hidden gold.",
            "- `very_sparse`, `under_extract`, `drug_hints_but_no_drug`, and `lab_values_but_no_results` point to likely recall loss and high WER.",
            "- `wide_candidates` points to candidate Jaccard loss; prefer fewer candidates when the mapping evidence is strong.",
            "- `suspicious_diagnosis_span` often means the model used report prose or imaging sections as diagnosis text.",
            "- `overlapping_same_type` should be reviewed manually because nested spans can be valid, but they often overcount symptoms.",
            "",
        ]
    )
    (review_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "file",
        "chars",
        "items",
        "per_1k",
        "diagnosis",
        "symptom",
        "drug",
        "test_name",
        "test_result",
        "isNegated",
        "isFamily",
        "isHistorical",
        "drug_hints",
        "lab_value_hints",
        "wide_candidates",
        "overlaps",
        "bad_candidates",
        "suspicious_diagnoses",
        "errors",
        "risk_score",
        "flags",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["file"])):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Format and review submitted output files.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--source-output", type=Path, default=ROOT / "output" / "output_10072026_llm_benchmark")
    parser.add_argument("--formatted-dir", type=Path, default=ROOT / "output" / "output_10072026_formatted")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "docs" / "review_output_10072026")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "data" / "candidates")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional path for a submission zip. Omit during manual review.",
    )
    args = parser.parse_args(argv)

    candidate_names = load_candidate_names(args.candidate_dir)
    args.formatted_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    input_files = sorted(args.input_dir.glob("*.txt"), key=sort_key)
    for input_path in input_files:
        fid = int(input_path.stem)
        output_path = args.source_output / f"{fid}.json"
        source_text = input_path.read_text(encoding="utf-8")
        payload = read_json(output_path) if output_path.exists() else []
        items = normalize_payload(payload)

        write_json(args.formatted_dir / f"{fid}.json", items)
        summary = audit_file(fid, source_text, items, candidate_names)
        rows.append(summary)
        write_review_note(
            args.review_dir / f"{fid:03d}.md",
            fid,
            source_text,
            items,
            summary,
            candidate_names,
        )

    if args.zip_path:
        package_zip(args.formatted_dir, args.zip_path)
    write_csv(args.review_dir / "summary.csv", rows)
    write_summary(args.review_dir, rows, args.source_output, args.formatted_dir, args.zip_path)
    print(f"formatted files: {args.formatted_dir}")
    print(f"review notes: {args.review_dir}")
    if args.zip_path:
        print(f"zip: {args.zip_path}")
    else:
        print("zip: not built")
    print(f"files: {len(rows)}")
    print(f"total_items: {sum(row['items'] for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
