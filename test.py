from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medkg.config import get_paths
from medkg.io import discover_input_files, read_text
from medkg.pipeline import MedicalKGPipeline


def post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def output_name(path: Path) -> str:
    if path.stem.isdigit():
        return f"{path.stem}.json"
    matches = re.findall(r"\d+", path.stem)
    return f"{matches[-1]}.json" if matches else "prediction.json"


def get_input_files(file_path: Path | None, input_dir: Path | None, legacy_path: Path | None) -> list[Path]:
    if file_path:
        return [file_path]
    if input_dir:
        return discover_input_files(input_dir)
    if legacy_path:
        return [legacy_path] if legacy_path.is_file() else discover_input_files(legacy_path)
    default_input = get_paths(ROOT).input_dir
    return discover_input_files(default_input)


def direct_predict(files: list[Path], mode: str) -> list[dict[str, Any]]:
    pipeline = MedicalKGPipeline(root=ROOT)
    results: list[dict[str, Any]] = []
    for path in files:
        concepts, meta = pipeline.process_text_with_meta(read_text(path), mode=mode)
        results.append({"id": path.stem, "concepts": [c.to_dict() for c in concepts], "meta": meta})
    return results


def server_predict(files: list[Path], mode: str, server_url: str) -> list[dict[str, Any]]:
    base = server_url.rstrip("/")
    if len(files) == 1:
        path = files[0]
        result = post_json(f"{base}/predict", {"id": path.stem, "text": read_text(path), "mode": mode, "validate": True})
        return [result]
    payload = {
        "mode": mode,
        "validate": True,
        "items": [{"id": path.stem, "text": read_text(path)} for path in files],
    }
    result = post_json(f"{base}/predict_batch", payload)
    results = result.get("results")
    if not isinstance(results, list):
        raise RuntimeError("server /predict_batch response has no results list")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual inference client for one .txt file or a folder")
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("baseline", "hybrid", "llm_full_doc"), default="hybrid")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pretty", action="store_true")
    # Backward-compatible aliases from the earlier baseline.
    parser.add_argument("--path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--server-url", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    out_dir = args.out or args.output_dir or (ROOT / "output")
    server_url = args.server_url or args.url
    files = get_input_files(args.file, args.input_dir, args.path)
    if args.limit is not None:
        files = files[: max(0, args.limit)]
    if not files:
        print("no input .txt files found")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        results = direct_predict(files, args.mode) if args.direct else server_predict(files, args.mode, server_url)
    except urllib.error.URLError as exc:
        print(f"cannot reach API at {server_url}: {exc}")
        print("start it with: python scripts/run_server.py")
        return 2

    fallback_count = 0
    llm_count = 0
    for path, result in zip(files, results):
        concepts = result.get("concepts")
        if not isinstance(concepts, list):
            raise RuntimeError(f"{path}: response has no concepts list")
        meta = result.get("meta") or {}
        fallback_count += 1 if meta.get("fallback_used") else 0
        llm_count += 1 if meta.get("llm_used") else 0
        out_path = out_dir / output_name(path)
        indent = 2 if args.pretty else None
        out_path.write_text(json.dumps(concepts, ensure_ascii=False, indent=indent), encoding="utf-8")

    summary = {
        "files": len(files),
        "output_dir": str(out_dir),
        "mode_requested": args.mode,
        "server_mode": not args.direct,
        "llm_used_files": llm_count,
        "fallback_files": fallback_count,
        "seconds": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
