from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.io import read_text
from core.schema import validate_output


@dataclass(frozen=True)
class FileResult:
    input_path: Path
    output_path: Path
    concepts: int
    llm_used: bool


def default_output_dir() -> Path:
    return ROOT / "output" / f"out_put_{datetime.now().strftime('%d%m%Y')}"


def discover_inputs(target: Path, limit: int | None) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".txt":
        files = [target]
    elif target.is_dir():
        files = sorted(
            target.glob("*.txt"),
            key=lambda path: (
                int(path.stem) if path.stem.isdigit() else 10**9,
                path.name,
            ),
        )
    else:
        raise ValueError(f"Khong tim thay file hoac thu muc input hop le: {target}")
    return files[: max(0, limit)] if limit is not None else files


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("API khong tra ve JSON object")
    return data


def run_file(
    input_path: Path,
    output_dir: Path,
    server_url: str,
    timeout: int,
    pretty: bool,
) -> FileResult:
    text = read_text(input_path)
    response = post_json(
        f"{server_url.rstrip('/')}/predict",
        {"text": text},
        timeout,
    )
    concepts = response.get("concepts")
    if not isinstance(concepts, list):
        raise ValueError(f"{input_path}: API khong tra ve concepts dang list")

    errors = validate_output(concepts, source_text=text)
    if errors:
        raise ValueError(f"{input_path}: {'; '.join(errors[:10])}")

    output_path = output_dir / f"{input_path.stem}.json"
    output_path.write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2 if pretty else None),
        encoding="utf-8",
    )
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    return FileResult(
        input_path=input_path,
        output_path=output_path,
        concepts=len(concepts),
        llm_used=bool(meta.get("llm_used")),
    )


def run_target(
    target: Path,
    output_dir: Path,
    server_url: str,
    workers: int,
    timeout: int,
    limit: int | None,
    pretty: bool,
) -> list[FileResult]:
    files = discover_inputs(target, limit)
    if not files:
        raise ValueError(f"Khong co file .txt trong {target}")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, min(workers, len(files)))
    results: list[FileResult] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_file,
                path,
                output_dir,
                server_url,
                timeout,
                pretty,
            ): path
            for path in files
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path}: {exc}")

    if failures:
        raise RuntimeError("\n".join(failures))
    return sorted(results, key=lambda item: item.input_path.name)


def main(argv: list[str] | None = None) -> int:
    default_workers = max(1, (os.cpu_count() or 1) * 2)
    parser = argparse.ArgumentParser(
        description="Chay input .txt song song qua local API server"
    )
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=ROOT / "input",
        help="Duong dan toi mot file .txt hoac thu muc chua cac file .txt",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    output_dir = (args.output_dir or default_output_dir()).resolve()
    try:
        results = run_target(
            args.target.resolve(),
            output_dir,
            args.url,
            args.workers,
            args.timeout,
            args.limit,
            args.pretty,
        )
    except urllib.error.URLError as exc:
        print(f"Khong ket noi duoc server {args.url}: {exc}")
        print("Khoi dong server bang: python main.py")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(str(exc))
        return 1

    summary = {
        "files": len(results),
        "concepts": sum(item.concepts for item in results),
        "llm_used_files": sum(item.llm_used for item in results),
        "workers": max(1, min(args.workers, len(results))),
        "output_dir": str(output_dir),
        "validation_errors": 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
