"""Run grouped OOF evaluation for the token proposal model only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_TYPES
from scripts.train_span_models import evaluate_token_oof, grouped_folds, load_documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate token proposal model with grouped folds.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input_part2" / "input" / "input")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "input_part2" / "gt" / "output")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "external" / "token_model_cv_report.json")
    args = parser.parse_args(argv)

    documents = load_documents(args.input_dir.resolve(), args.gold_dir.resolve())
    folds = grouped_folds(documents, max(2, args.folds))
    labels = ("O",) + tuple(
        label
        for concept_type in ALLOWED_TYPES
        for label in (f"B:{concept_type}", f"I:{concept_type}")
    )
    scores = evaluate_token_oof(documents, folds, labels, epochs=max(1, args.epochs))
    payload = {
        "documents": len(documents),
        "folds": len(folds),
        "epochs": max(1, args.epochs),
        "exact_span_and_type": scores,
    }
    with args.report.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
