from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import SYSTEM_RXNORM, get_paths
from knowledge.ontology import seed_entries


RXNAV = "https://rxnav.nlm.nih.gov/REST"


REQUIRED_TERMS = [
    "amlodipine",
    "aspirin",
    "metoprolol succinate",
    "guaifenesin",
    "nystatin",
    "acetaminophen",
    "pravastatin",
    "docusate sodium",
    "senna",
    "clonazepam",
    "chlorpheniramine",
    "capsaicin",
    "levophed",
    "norepinephrine",
    "propofol",
    "phentolamine",
]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def fallback_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in seed_entries():
        if entry.system != SYSTEM_RXNORM:
            continue
        rows.append(
            {
                "rxcui": entry.code,
                "canonical_name": entry.name,
                "aliases": list(entry.aliases),
                "tty": "",
                "source_system": "RXNORM",
                "source": "manual_seed",
            }
        )
    return rows


def query_rxnav(term: str, cache_dir: Path, timeout: int) -> list[dict[str, object]]:
    cache_path = cache_dir / f"{urllib.parse.quote(term, safe='').replace('%', '_')}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        url = f"{RXNAV}/drugs.json?" + urllib.parse.urlencode({"name": term})
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, object]] = []
    for group in data.get("drugGroup", {}).get("conceptGroup", []):
        tty = group.get("tty") or ""
        for concept in group.get("conceptProperties", []) or []:
            rxcui = str(concept.get("rxcui") or "")
            name = str(concept.get("name") or "")
            synonym = str(concept.get("synonym") or "")
            aliases = [term, name]
            if synonym:
                aliases.append(synonym)
            if rxcui and name:
                rows.append(
                    {
                        "rxcui": rxcui,
                        "canonical_name": name,
                        "aliases": sorted(set(a for a in aliases if a)),
                        "tty": tty,
                        "source_system": "RXNORM",
                        "source": "rxnav_api",
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Download or seed RxNorm/RxNav drug concepts")
    parser.add_argument("--terms-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args(argv)

    terms = REQUIRED_TERMS
    if args.terms_file:
        terms = [line.strip() for line in args.terms_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        terms = terms[: args.limit]

    cache_dir = paths.data_raw / "rxnorm" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for term in terms:
        try:
            rows.extend(query_rxnav(term, cache_dir, args.timeout))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{term}: {exc}")
    if not rows:
        rows = fallback_rows()
    else:
        rows.extend(fallback_rows())
    dedup: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row.get("rxcui") or row.get("canonical_name"))
        if key not in dedup:
            dedup[key] = row
        else:
            aliases = set(dedup[key].get("aliases") or [])
            aliases.update(row.get("aliases") or [])
            dedup[key]["aliases"] = sorted(aliases)
    out_path = paths.data_processed / "rxnorm" / "rxnorm_concepts.jsonl"
    write_jsonl(out_path, list(dedup.values()))
    status = {
        "source": "rxnorm",
        "downloaded": len(failures) < len(terms),
        "rows": len(dedup),
        "failures": failures,
    }
    (paths.data_processed / "rxnorm" / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out_path), **status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
