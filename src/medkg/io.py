"""Input discovery and output writing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from medkg.schema import Concept


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group(0)) if match else 10**9, path.name)


def discover_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [p for p in input_dir.rglob("*.txt") if p.is_file()]
    return sorted(files, key=natural_key)


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}.json"


def write_output(path: Path, concepts: Iterable[Concept]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [concept.to_dict() for concept in concepts]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

