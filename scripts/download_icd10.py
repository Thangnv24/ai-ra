from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import SYSTEM_ICD10, TYPE_DIAGNOSIS, get_paths
from medkg.ontology import seed_entries


ICD10_CM_FY26_APRIL_ZIP = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/"
    "ICD10CM/2026-update/icd10cm-Code%20Descriptions-April-1-2026.zip"
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def fallback_rows(source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in seed_entries():
        if entry.system != SYSTEM_ICD10:
            continue
        rows.append(
            {
                "code": entry.code,
                "canonical_name": entry.name,
                "aliases": list(entry.aliases),
                "source_system": SYSTEM_ICD10,
                "source": source,
            }
        )
    return rows


def parse_icd_zip(blob: bytes) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        txt_files = [name for name in zf.namelist() if name.lower().endswith(".txt")]
        if not txt_files:
            raise RuntimeError("No .txt code-description file found in ICD-10 ZIP")
        with zf.open(txt_files[0]) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    rows.append(
                        {
                            "code": parts[0],
                            "canonical_name": parts[1],
                            "aliases": [parts[1]],
                            "source_system": SYSTEM_ICD10,
                            "source": "cdc_icd10cm_fy26_apr2026",
                        }
                    )
    return rows


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Download or seed ICD-10-CM concept descriptions")
    parser.add_argument("--url", default=ICD10_CM_FY26_APRIL_ZIP)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args(argv)

    raw_dir = paths.data_raw / "icd10"
    out_path = paths.data_processed / "icd10" / "icd10_concepts.jsonl"
    status = {"source": "icd10", "downloaded": False, "rows": 0, "error": None}
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:  # noqa: S310
            blob = response.read()
        (raw_dir / "icd10cm_code_descriptions.zip").write_bytes(blob)
        rows = parse_icd_zip(blob)
        status["downloaded"] = True
    except Exception as exc:  # noqa: BLE001
        rows = fallback_rows("manual_seed_fallback_after_icd10_download_failure")
        status["error"] = str(exc)
    write_jsonl(out_path, rows)
    status["rows"] = len(rows)
    status_path = paths.data_processed / "icd10" / "status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), **status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
