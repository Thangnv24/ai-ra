from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Cach chay tu thu muc goc du an:
#   python main.py
#   python tests/test.py
# Mac dinh lenh tren doc full folder input/ va ghi output vao:
#   output/out_put_DDMMYYYY/1.json, output/out_put_DDMMYYYY/2.json, ...
# Chay rieng 1 file txt:
#   python tests/test.py input/1.txt
# File tren se di qua API /predict va ghi ra:
#   output/out_put_DDMMYYYY/1.json
# Neu muon chi dinh thu muc output rieng:
#   python tests/test.py input/1.txt --output-dir output/single_run
# Khi do ket qua ghi ra:
#   output/single_run/1.json

from core.schema import validate_output
from services.pipeline import MedicalKGPipeline


@dataclass(frozen=True)
class FileResult:
    input_path: Path
    output_path: Path
    concepts: int
    llm_used: bool


def default_output_dir() -> Path:
    return ROOT / "output" / f"out_put_{datetime.now().strftime('%d%m%Y')}"


def discover_inputs(target: Path) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".txt":
        return [target]
    if target.is_dir():
        return sorted(
            target.glob("*.txt"),
            key=lambda path: (
                int(path.stem) if path.stem.isdigit() else 10**9,
                path.name,
            ),
        )
    raise ValueError(f"Khong tim thay file hoac thu muc input hop le: {target}")


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
    mode: str,
    timeout: int,
) -> FileResult:
    text = input_path.read_text(encoding="utf-8-sig")
    response = post_json(
        f"{server_url.rstrip('/')}/predict",
        {"id": input_path.stem, "text": text, "mode": mode, "validate": True},
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
        json.dumps(concepts, ensure_ascii=False, indent=2),
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
    mode: str,
    workers: int,
    timeout: int,
) -> list[FileResult]:
    files = discover_inputs(target)
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
                mode,
                timeout,
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


class EndToEndFlowTest(unittest.TestCase):
    def test_direct_pipeline_returns_schema_valid_output(self) -> None:
        text = (
            "Benh nhan khong ho. Co tien su hen suyen. "
            "Dang dung aspirin 81 mg po daily. WBC: 12 mg/dL"
        )
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir()
            (input_dir / "1.txt").write_text(text, encoding="utf-8")

            summary = MedicalKGPipeline(root=ROOT).run_directory(
                input_dir,
                output_dir,
                limit=1,
            )
            payload = json.loads(
                (output_dir / "1.json").read_text(encoding="utf-8-sig")
            )
            self.assertEqual(summary.files, 1)
            self.assertEqual(validate_output(payload, source_text=text), [])


def main(argv: list[str] | None = None) -> int:
    default_workers = max(1, (os.cpu_count() or 1) * 2)
    parser = argparse.ArgumentParser(
        description="Chay E2E qua server cho mot file .txt hoac ca thu muc"
    )
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=ROOT / "input",
        help="Duong dan toi mot file .txt hoac thu muc chua cac file .txt",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--mode",
        choices=("baseline", "hybrid", "llm_full_doc"),
        default="hybrid",
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    try:
        results = run_target(
            args.target.resolve(),
            args.output_dir.resolve(),
            args.url,
            args.mode,
            args.workers,
            args.timeout,
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
        "output_dir": str(args.output_dir.resolve()),
        "validation_errors": 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
