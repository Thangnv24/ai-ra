from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


STEPS = [
    ["scripts/download_icd10.py"],
    ["scripts/download_rxnorm.py"],
    ["scripts/download_public_corpora.py"],
    ["scripts/prepare_vi_medical_aliases.py"],
    ["scripts/inspect_data_status.py"],
]


def main() -> int:
    results: list[dict[str, object]] = []
    for step in STEPS:
        cmd = [sys.executable, *step]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        results.append(
            {
                "command": " ".join(step),
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip()[-2000:],
                "stderr": proc.stderr.strip()[-2000:],
            }
        )
        print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
    out_path = ROOT / "data" / "processed" / "download_all_status.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
