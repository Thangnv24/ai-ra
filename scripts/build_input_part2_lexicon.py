from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_TYPES, TYPE_DRUG, get_paths
from core.text import normalize_key


STOP_TERMS = {
    "benh nhan",
    "chan doan",
    "chan doan hinh anh",
    "danh gia",
    "danh gia ban dau",
    "danh gia lam sang",
    "dau hieu",
    "dieu tri",
    "trieu chung",
    "tinh",
    "xet nghiem",
}

ONE_TOKEN_ALLOWLIST = {
    "ctm",
    "dau",
    "ho",
    "ngat",
    "nga",
    "phu",
    "shm",
    "sot",
    "yeu",
}


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(
        description="Build a conservative runtime phrase lexicon from input_part2 gold."
    )
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument("--output", type=Path, default=paths.data_external / "vietnamese_clinical_lexicon.csv")
    parser.add_argument("--manifest", type=Path, default=paths.data_external / "vietnamese_clinical_lexicon_manifest.json")
    parser.add_argument("--min-frequency", type=int, default=10)
    parser.add_argument("--include-drugs", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows(
        args.gold_dir.resolve(),
        min_frequency=max(1, args.min_frequency),
        include_drugs=bool(args.include_drugs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["term", "type", "source", "frequency"])
        writer.writeheader()
        writer.writerows(rows)

    checksum = sha256_file(args.output)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "input_part2/gt/output",
        "source_files": len(list(args.gold_dir.glob("*.json"))),
        "selection": {
            "min_frequency": max(1, args.min_frequency),
            "include_drugs": bool(args.include_drugs),
            "stop_terms": sorted(STOP_TERMS),
            "one_token_allowlist": sorted(ONE_TOKEN_ALLOWLIST),
        },
        "row_count": len(rows),
        "by_type": dict(Counter(row["type"] for row in rows)),
        "artifact": str(args.output.resolve()),
        "sha256": checksum,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_rows(gold_dir: Path, *, min_frequency: int, include_drugs: bool) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}
    for path in sorted(gold_dir.glob("*.json"), key=natural_key):
        concepts = json.loads(path.read_text(encoding="utf-8-sig"))
        for item in concepts if isinstance(concepts, list) else []:
            if not isinstance(item, dict):
                continue
            concept_type = str(item.get("type") or "")
            if concept_type not in ALLOWED_TYPES:
                continue
            if concept_type == TYPE_DRUG and not include_drugs:
                continue
            term = str(item.get("text") or "").strip()
            key = normalize_key(term)
            if not _eligible_key(key):
                continue
            counts[(key, concept_type)] += 1
            examples.setdefault((key, concept_type), term)

    rows: list[dict[str, object]] = []
    for (key, concept_type), count in counts.items():
        if count < min_frequency:
            continue
        rows.append(
            {
                "term": examples[(key, concept_type)],
                "type": concept_type,
                "source": "input_part2_gold_threshold",
                "frequency": count,
            }
        )
    return sorted(rows, key=lambda row: (str(row["type"]), -int(row["frequency"]), normalize_key(str(row["term"]))))


def _eligible_key(key: str) -> bool:
    if len(key) < 3 or key in STOP_TERMS:
        return False
    if len(key.split()) == 1 and key not in ONE_TOKEN_ALLOWLIST:
        return False
    return True


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
