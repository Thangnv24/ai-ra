"""Create the competition submission archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from core.config import get_paths


def package_submission(output_dir: Path, submission_dir: Path) -> Path:
    submission_dir.mkdir(parents=True, exist_ok=True)
    zip_path = submission_dir / "output.zip"
    files = sorted(output_dir.glob("*.json"), key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.name))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=f"output/{path.name}")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Package outputs into submission/output.zip")
    parser.add_argument("--output-dir", type=Path, default=paths.output_dir)
    parser.add_argument("--submission-dir", type=Path, default=paths.submission_dir)
    args = parser.parse_args(argv)
    zip_path = package_submission(args.output_dir, args.submission_dir)
    count = len(list(args.output_dir.glob("*.json"))) if args.output_dir.exists() else 0
    print(f"created {zip_path} with {count} json files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
