from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import get_paths
from medkg.io import load_json, read_text
from medkg.schema import validate_output


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Validate output JSON files against the competition schema")
    parser.add_argument("--output-dir", type=Path, default=paths.output_dir)
    parser.add_argument("--input-dir", type=Path, default=paths.input_dir)
    args = parser.parse_args(argv)

    files = sorted(args.output_dir.glob("*.json"), key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.name)) if args.output_dir.exists() else []
    all_errors: list[str] = []
    for path in files:
        source_path = args.input_dir / f"{path.stem}.txt"
        source_text = read_text(source_path) if source_path.exists() else None
        try:
            payload = load_json(path)
        except Exception as exc:  # noqa: BLE001
            all_errors.append(f"{path}: invalid JSON: {exc}")
            continue
        errors = validate_output(payload, source_text=source_text)
        all_errors.extend(f"{path}: {err}" for err in errors)

    result = {"files": len(files), "errors": len(all_errors)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for error in all_errors[:50]:
        print(error)
    if len(all_errors) > 50:
        print(f"... {len(all_errors) - 50} more errors")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

