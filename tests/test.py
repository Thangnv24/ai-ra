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
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.io import read_text
from core.schema import validate_output
from services.pipeline import MedicalKGPipeline


@dataclass(frozen=True)
class FileResult:
    input_path: Path
    output_path: Path
    concepts: int
    llm_used: bool


@dataclass(frozen=True)
class CompareFileResult:
    input_path: Path
    without_final_output_path: Path
    with_final_output_path: Path
    diff_path: Path
    without_final_concepts: int
    with_final_concepts: int
    added_by_final: int
    removed_by_final: int
    changed_same_span: int


def default_output_dir() -> Path:
    return ROOT / "output" / f"out_put_{datetime.now().strftime('%d%m%Y')}"


def default_compare_output_dir() -> Path:
    return ROOT / "output" / f"compare_final_llm_{datetime.now().strftime('%d%m%Y_%H%M%S')}"


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


def run_compare_final_llm(
    target: Path,
    output_dir: Path,
    limit: int | None,
    pretty: bool,
) -> list[CompareFileResult]:
    files = discover_inputs(target, limit)
    if not files:
        raise ValueError(f"Khong co file .txt trong {target}")

    without_dir = output_dir / "without_final_llm"
    with_dir = output_dir / "with_final_llm"
    diff_dir = output_dir / "diff"
    without_dir.mkdir(parents=True, exist_ok=True)
    with_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)

    pipeline = MedicalKGPipeline(root=ROOT)
    results: list[CompareFileResult] = []
    for input_path in files:
        text = read_text(input_path)
        without_concepts = _process_text_without_final_llm(pipeline, text)
        with_concepts = _process_text_with_final_llm(pipeline, text)

        without_output_path = without_dir / f"{input_path.stem}.json"
        with_output_path = with_dir / f"{input_path.stem}.json"
        diff_path = diff_dir / f"{input_path.stem}.json"
        _write_concepts(without_output_path, without_concepts, pretty)
        _write_concepts(with_output_path, with_concepts, pretty)
        diff = _compare_concepts(without_concepts, with_concepts)
        diff_path.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "without_final_llm": str(without_output_path),
                    "with_final_llm": str(with_output_path),
                    **diff,
                },
                ensure_ascii=False,
                indent=2 if pretty else None,
            ),
            encoding="utf-8",
        )
        results.append(
            CompareFileResult(
                input_path=input_path,
                without_final_output_path=without_output_path,
                with_final_output_path=with_output_path,
                diff_path=diff_path,
                without_final_concepts=len(without_concepts),
                with_final_concepts=len(with_concepts),
                added_by_final=len(diff["added_by_final_llm"]),
                removed_by_final=len(diff["removed_by_final_llm"]),
                changed_same_span=len(diff["changed_same_span"]),
            )
        )
    return results


def _process_text_with_final_llm(pipeline: MedicalKGPipeline, text: str) -> list[dict[str, Any]]:
    concepts, _ = pipeline.process_text_with_meta(text)
    output = [concept.to_dict() for concept in concepts]
    _validate_direct_output(output, text)
    return output


def _process_text_without_final_llm(pipeline: MedicalKGPipeline, text: str) -> list[dict[str, Any]]:
    original_apply = pipeline._apply_llm_decisions
    pipeline._apply_llm_decisions = MethodType(_skip_final_llm_decisions, pipeline)
    try:
        concepts, _ = pipeline.process_text_with_meta(text)
    finally:
        pipeline._apply_llm_decisions = original_apply
    output = [concept.to_dict() for concept in concepts]
    _validate_direct_output(output, text)
    return output


def _skip_final_llm_decisions(
    self: MedicalKGPipeline,
    text: str,
    concepts: list[Any],
    meta: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    # Compare mode only: skip the final LLM decision/rerank pass while keeping
    # the required first LLM entity extraction unchanged.
    meta["mode_used"] = "llm_full_doc"
    meta["llm_used"] = True
    meta["llm_decisions"] = 0
    meta["llm_decision_scope"] = "disabled_by_tests_compare_final_llm"
    meta["llm_decision_passthrough"] = len(concepts)
    return sorted(concepts, key=lambda concept: (concept.position[0], concept.position[1], concept.type)), meta


def _validate_direct_output(concepts: list[dict[str, Any]], text: str) -> None:
    errors = validate_output(concepts, source_text=text)
    if errors:
        raise ValueError("; ".join(errors[:10]))


def _write_concepts(output_path: Path, concepts: list[dict[str, Any]], pretty: bool) -> None:
    output_path.write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2 if pretty else None),
        encoding="utf-8",
    )


def _compare_concepts(without_final: list[dict[str, Any]], with_final: list[dict[str, Any]]) -> dict[str, Any]:
    without_by_key = {_concept_key(item): item for item in without_final}
    with_by_key = {_concept_key(item): item for item in with_final}
    without_span_map = {_span_text_key(item): item for item in without_final}
    with_span_map = {_span_text_key(item): item for item in with_final}
    added_keys = sorted(set(with_by_key) - set(without_by_key))
    removed_keys = sorted(set(without_by_key) - set(with_by_key))
    changed: list[dict[str, Any]] = []
    for key in sorted(set(without_span_map) & set(with_span_map)):
        before = without_span_map[key]
        after = with_span_map[key]
        if before != after:
            changed.append({"without_final_llm": before, "with_final_llm": after})
    return {
        "counts": {
            "without_final_llm": len(without_final),
            "with_final_llm": len(with_final),
            "added_by_final_llm": len(added_keys),
            "removed_by_final_llm": len(removed_keys),
            "changed_same_span": len(changed),
        },
        "added_by_final_llm": [with_by_key[key] for key in added_keys],
        "removed_by_final_llm": [without_by_key[key] for key in removed_keys],
        "changed_same_span": changed,
    }


def _concept_key(item: dict[str, Any]) -> tuple[Any, ...]:
    position = item.get("position") if isinstance(item.get("position"), list) else []
    return (tuple(position), item.get("text"), item.get("type"), tuple(item.get("candidates") or ()))


def _span_text_key(item: dict[str, Any]) -> tuple[Any, ...]:
    position = item.get("position") if isinstance(item.get("position"), list) else []
    return (tuple(position), item.get("text"))


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
    # 1. Chay binh thuong qua API server, co LLM theo pipeline hien tai.
    #    Nghia la van co LLM trich xuat entity; LLM cuoi chi chay neu pipeline thay can.
    #    Can khoi dong server truoc bang: python main.py
    #    Vi du 1 file:
    #      python tests/test.py input/1.txt --pretty
    #    Vi du ca thu muc input:
    #      python tests/test.py input --pretty
    #
    # 2. Chay so sanh truc tiep "khong LLM cuoi" va "co LLM cuoi".
    #    Lenh nay khong gui mode vao /predict, ma chay pipeline truc tiep trong test runner.
    #    - Ket qua khong LLM cuoi nam trong: <output-dir>/without_final_llm/
    #    - Ket qua co LLM cuoi nam trong:    <output-dir>/with_final_llm/
    #    - Diff nam trong:                  <output-dir>/diff/
    #    Luu y: ca hai nhanh van giu LLM trich xuat entity dau tien.
    #    Vi du:
    #      python tests/test.py input/1.txt --compare-final-llm --pretty
    #
    # 3. Chay qua API server voi so worker tuy y.
    #    --workers chi ap dung cho mode chay qua API server, khong ap dung cho --compare-final-llm.
    #    Vi du chay 1 worker de giam tai LiteLLM proxy:
    #      python tests/test.py input --workers 1 --timeout 600
    #    Vi du chay 8 worker khi endpoint du khoe:
    #      python tests/test.py input --workers 8 --timeout 600
    parser.add_argument(
        "--compare-final-llm",
        action="store_true",
        help="Chay moi file 2 lan de so sanh khong LLM cuoi va co LLM cuoi.",
    )
    args = parser.parse_args(argv)

    output_dir = (
        args.output_dir
        or (default_compare_output_dir() if args.compare_final_llm else default_output_dir())
    ).resolve()
    try:
        if args.compare_final_llm:
            compare_results = run_compare_final_llm(
                args.target.resolve(),
                output_dir,
                args.limit,
                args.pretty,
            )
            summary = {
                "mode": "compare_final_llm",
                "files": len(compare_results),
                "without_final_llm_concepts": sum(item.without_final_concepts for item in compare_results),
                "with_final_llm_concepts": sum(item.with_final_concepts for item in compare_results),
                "added_by_final_llm": sum(item.added_by_final for item in compare_results),
                "removed_by_final_llm": sum(item.removed_by_final for item in compare_results),
                "changed_same_span": sum(item.changed_same_span for item in compare_results),
                "output_dir": str(output_dir),
                "without_final_llm_dir": str(output_dir / "without_final_llm"),
                "with_final_llm_dir": str(output_dir / "with_final_llm"),
                "diff_dir": str(output_dir / "diff"),
                "validation_errors": 0,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
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
