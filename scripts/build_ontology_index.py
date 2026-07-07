from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import get_paths
from medkg.ontology import build_ontology_index


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Build local ICD-10/RxNorm ontology index")
    parser.add_argument("--raw-dir", type=Path, default=paths.data_raw)
    parser.add_argument("--external-dir", type=Path, default=paths.data_external)
    parser.add_argument("--output", type=Path, default=paths.index_file)
    args = parser.parse_args(argv)

    index = build_ontology_index(args.raw_dir, args.external_dir, args.output)
    systems: dict[str, int] = {}
    for entry in index.entries:
        systems[entry.system] = systems.get(entry.system, 0) + 1
    print(json.dumps({"output": str(args.output), "entries": len(index.entries), "systems": systems}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

