from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.io import read_text
from core.schema import validate_output


RUN_START = time.perf_counter()
LOG_LOCK = Lock()


@dataclass(frozen=True)
class FileResult:
    input_path: Path
    output_path: Path
    concepts: int
    llm_used: bool


def default_output_dir(target: Path | None = None) -> Path:
    name = f"out_put_{datetime.now().strftime('%d%m%Y')}"
    if target is not None and "input_part2" in {part.casefold() for part in target.resolve().parts}:
        name += "_part2"
    return ROOT / "output" / name


def log_step(message: str, step_start: float | None = None) -> None:
    now = time.perf_counter()
    elapsed = now - RUN_START
    duration = f" duration={now - step_start:.2f}s" if step_start is not None else ""
    with LOG_LOCK:
        print(f"[{datetime.now().strftime('%H:%M:%S')} +{elapsed:.2f}s] {message}{duration}", flush=True)


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
    file_start = time.perf_counter()
    log_step(f"api_file_start file={input_path.name} final_llm=removed")

    read_start = time.perf_counter()
    text = read_text(input_path)
    log_step(f"api_read_done file={input_path.name} chars={len(text)}", read_start)

    api_start = time.perf_counter()
    log_step(f"api_request_start file={input_path.name} url={server_url.rstrip('/')}/predict timeout={timeout}s")
    response = post_json(
        f"{server_url.rstrip('/')}/predict",
        {"text": text},
        timeout,
    )
    log_step(f"api_response_done file={input_path.name}", api_start)

    concepts = response.get("concepts")
    if not isinstance(concepts, list):
        raise ValueError(f"{input_path}: API khong tra ve concepts dang list")

    validate_start = time.perf_counter()
    errors = validate_output(concepts, source_text=text)
    if errors:
        raise ValueError(f"{input_path}: {'; '.join(errors[:10])}")
    log_step(f"api_validate_done file={input_path.name} concepts={len(concepts)}", validate_start)

    output_path = output_dir / f"{input_path.stem}.json"
    write_start = time.perf_counter()
    output_path.write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2 if pretty else None),
        encoding="utf-8",
    )
    log_step(f"api_write_done file={input_path.name} output={output_path}", write_start)

    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    llm_used = bool(meta.get("llm_used"))
    log_step(
        f"api_file_done file={input_path.name} concepts={len(concepts)} "
        f"entity_llm_used={llm_used} final_llm_used={bool(meta.get('final_llm_used'))} "
        f"output={output_path}",
        file_start,
    )
    return FileResult(
        input_path=input_path,
        output_path=output_path,
        concepts=len(concepts),
        llm_used=llm_used,
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
    run_start = time.perf_counter()
    files = discover_inputs(target, limit)
    if not files:
        raise ValueError(f"Khong co file .txt trong {target}")

    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete_marker = output_dir / "_INCOMPLETE_RUN.txt"
    incomplete_marker.write_text(
        f"Run started with {len(files)} expected files. Do not submit this folder until this marker is removed.\n",
        encoding="utf-8",
    )
    max_workers = max(1, min(workers, len(files)))
    log_step(
        f"api_run_start target={target} files={len(files)} workers={max_workers} "
        f"entity_llm=enabled final_llm=removed output_dir={output_dir}"
    )

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
                log_step(f"api_file_failed file={path.name} error={exc}")
                failures.append(f"{path}: {exc}")

    if failures:
        incomplete_marker.write_text(
            "Run failed. Missing or failed requests:\n" + "\n".join(failures) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("\n".join(failures))

    expected_names = {f"{path.stem}.json" for path in files}
    result_names = {item.output_path.name for item in results}
    disk_names = {path.name for path in output_dir.glob("*.json")}
    missing_results = expected_names - result_names
    missing_disk = expected_names - disk_names
    unexpected_disk = disk_names - expected_names if target.is_dir() and limit is None else set()
    if missing_results or missing_disk or unexpected_disk:
        details = (
            f"Incomplete output set: missing_results={sorted(missing_results)} "
            f"missing_disk={sorted(missing_disk)} unexpected_disk={sorted(unexpected_disk)}"
        )
        incomplete_marker.write_text(details + "\n", encoding="utf-8")
        raise RuntimeError(details)
    incomplete_marker.unlink()
    log_step(
        f"api_run_done files={len(results)} concepts={sum(item.concepts for item in results)} "
        f"output_dir={output_dir}",
        run_start,
    )
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
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pretty", action="store_true")

    # CAC LENH THUONG DUNG:
    #
    # Pipeline chi con mot luong: LLM trich xuat entity lan dau, sau do map
    # deterministic; khong con LLM decision/rerank lan hai.
    # Khoi dong API server truoc bang:
    #   python main.py
    #
    # Chay toan bo input voi cau hinh mac dinh:
    #   python tests/test.py
    #
    # Chay mot file:
    #   python tests/test.py input/1.txt --pretty
    #
    # Chay voi so worker tuy chon (nen dung 1 khi proxy cham):
    #   python tests/test.py input --workers 1 --timeout 600
    #   python tests/test.py input --workers 8 --timeout 600
    args = parser.parse_args(argv)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else default_output_dir(args.target).resolve()
    )
    log_step(
        f"runner_start mode=entity_llm_only target={args.target.resolve()} "
        f"output_dir={output_dir}"
    )
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
        "mode": "entity_llm_only",
        "files": len(results),
        "concepts": sum(item.concepts for item in results),
        "entity_llm_used_files": sum(item.llm_used for item in results),
        "final_llm": False,
        "workers": max(1, min(args.workers, len(results))),
        "output_dir": str(output_dir),
        "validation_errors": 0,
    }
    log_step(f"api_summary concepts={summary['concepts']} output_dir={output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
