from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import get_paths


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Record optional public corpus status")
    parser.add_argument("--attempt", action="store_true", help="reserved for future dataset downloads")
    parser.parse_args(argv)
    status = {
        "source": "public_corpora",
        "downloaded": False,
        "resources": {
            "NCBI Disease Corpus": "optional; not downloaded by default in this reproducible flow",
            "MedMentions": "optional; not downloaded by default in this reproducible flow",
            "BC5CDR": "optional; not downloaded by default in this reproducible flow",
        },
        "note": "These corpora are development resources and do not block inference.",
    }
    out_path = paths.data_raw / "public_corpora" / "status.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), **status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
