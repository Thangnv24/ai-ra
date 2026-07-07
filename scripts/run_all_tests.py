from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import get_paths
from medkg.pipeline import MedicalKGPipeline


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Run deterministic inference on all test input files")
    parser.add_argument("--input-dir", type=Path, default=paths.input_dir)
    parser.add_argument("--output-dir", type=Path, default=paths.output_dir)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    pipeline = MedicalKGPipeline(root=ROOT)
    summary = pipeline.run_directory(args.input_dir, args.output_dir, limit=args.limit)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    if summary.files == 0:
        print(f"warning: no .txt files found under {args.input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

