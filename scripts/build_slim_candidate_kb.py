from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import SYSTEM_ICD10, SYSTEM_RXNORM, TYPE_DIAGNOSIS, TYPE_DRUG
from core.text import normalize_key


RXNORM_KEEP_SABS = {"RXNORM"}
RXNORM_KEEP_TTYS = {
    "BN",
    "BPCK",
    "DF",
    "DFG",
    "GPCK",
    "IN",
    "MIN",
    "PIN",
    "PSN",
    "SBD",
    "SBDG",
    "SBDF",
    "SBDFP",
    "SBDC",
    "SCD",
    "SCDC",
    "SCDG",
    "SCDF",
    "SCDFP",
    "SY",
    "TMSY",
}
TTY_PRIORITY = {
    "SCD": 1,
    "SBD": 1,
    "SBDC": 2,
    "SCDC": 2,
    "SCDF": 2,
    "SBDF": 2,
    "SCDG": 3,
    "SBDG": 3,
    "IN": 5,
    "PIN": 5,
    "MIN": 6,
    "BN": 7,
    "PSN": 8,
    "SY": 9,
    "TMSY": 9,
    "DF": 20,
    "DFG": 20,
    "GPCK": 30,
    "BPCK": 30,
}


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = " ".join(str(value or "").split())
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def alias_norms(values: list[str]) -> list[str]:
    return unique_nonempty([normalize_key(value) for value in values])


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in read_jsonl(path)]


def icd_priority(row: dict[str, Any]) -> int:
    code = str(row.get("disease_code") or "")
    if row.get("flag_not_primary") or row.get("flag_not_recommended_primary"):
        return 80
    if row.get("flag_not_used_because_more_specific") == code:
        return 40
    if "." in code:
        return 10
    return 20


def build_icd10_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    for item in read_jsonl(path):
        code = str(item.get("disease_code") or "").strip()
        if not code:
            continue
        aliases = unique_nonempty(
            [
                str(item.get("disease_name_vi") or ""),
                str(item.get("disease_name_en") or ""),
                str(item.get("three_char_name_vi") or ""),
                str(item.get("three_char_name_en") or ""),
                str(item.get("additional_guidance_vi") or ""),
                str(item.get("additional_guidance_en") or ""),
                str(item.get("disease_name_vi_norm") or ""),
                str(item.get("disease_name_en_norm") or ""),
                str(item.get("three_char_name_vi_norm") or ""),
                str(item.get("three_char_name_en_norm") or ""),
            ]
        )
        rows.append(
            {
                "code": code,
                "code_nodot": str(item.get("disease_code_nodot") or "").strip(),
                "system": SYSTEM_ICD10,
                "type": TYPE_DIAGNOSIS,
                "name_vi": str(item.get("disease_name_vi") or "").strip(),
                "name_en": str(item.get("disease_name_en") or "").strip(),
                "aliases": aliases,
                "alias_norms": alias_norms(aliases),
                "priority": icd_priority(item),
            }
        )
    return rows


def rxnorm_name_rank(tty: str, is_pref: str, value: str) -> tuple[int, int, int, str]:
    return (
        TTY_PRIORITY.get(tty, 99),
        0 if is_pref == "Y" else 1,
        len(value),
        value.casefold(),
    )


def add_rxnorm_grouped_row(
    grouped: dict[str, dict[str, Any]],
    rxcui: str,
    name: str,
    tty: str,
    is_pref: str = "",
) -> None:
    rxcui = rxcui.strip()
    name = " ".join(name.split())
    if not rxcui or not name or tty not in RXNORM_KEEP_TTYS:
        return
    row = grouped.setdefault(
        rxcui,
        {
            "rxcui": rxcui,
            "system": SYSTEM_RXNORM,
            "type": TYPE_DRUG,
            "name": name,
            "name_rank": rxnorm_name_rank(tty, is_pref, name),
            "aliases": set(),
            "ttys": set(),
            "priority": TTY_PRIORITY.get(tty, 99),
            "archive": False,
        },
    )
    row["aliases"].add(name)
    row["ttys"].add(tty)
    row["priority"] = min(int(row["priority"]), TTY_PRIORITY.get(tty, 99))
    rank = rxnorm_name_rank(tty, is_pref, name)
    if rank < row["name_rank"]:
        row["name"] = name
        row["name_rank"] = rank


def add_rxnorm_conso_rows(grouped: dict[str, dict[str, Any]], path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue
            rxcui = parts[0].strip()
            lat = parts[1].strip()
            is_pref = parts[6].strip()
            sab = parts[11].strip()
            tty = parts[12].strip()
            name = " ".join(parts[14].split())
            suppress = parts[16].strip()
            if (
                not rxcui
                or not name
                or lat != "ENG"
                or sab not in RXNORM_KEEP_SABS
                or tty not in RXNORM_KEEP_TTYS
                or suppress not in {"N", "O", ""}
            ):
                continue
            add_rxnorm_grouped_row(grouped, rxcui, name, tty, is_pref)


def add_rxnorm_archive_rows(grouped: dict[str, dict[str, Any]], path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue
            name = " ".join(parts[2].split())
            rxcui = parts[6].strip()
            lat = parts[8].strip()
            sab = parts[13].strip()
            tty = parts[14].strip()
            if (
                not rxcui
                or not name
                or lat != "ENG"
                or sab not in RXNORM_KEEP_SABS
                or tty not in RXNORM_KEEP_TTYS
            ):
                continue
            add_rxnorm_grouped_row(grouped, rxcui, name, tty)
            grouped[rxcui]["archive"] = True


def build_rxnorm_rows(conso_path: Path, archive_path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    add_rxnorm_conso_rows(grouped, conso_path)
    add_rxnorm_archive_rows(grouped, archive_path)
    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        aliases = sorted(row["aliases"], key=lambda value: (normalize_key(value), value))
        # Keep the artifact compact while preserving useful synonyms.
        aliases = aliases[:40]
        rows.append(
            {
                "code": row["rxcui"],
                "system": SYSTEM_RXNORM,
                "type": TYPE_DRUG,
                "name": row["name"],
                "aliases": aliases,
                "alias_norms": alias_norms(aliases),
                "ttys": sorted(row["ttys"]),
                "priority": int(row["priority"]),
                "archive": bool(row.get("archive")),
            }
        )
    return sorted(rows, key=lambda item: (item["priority"], item["code"]))


def compact_rxnorm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep lookup aliases in candidate_aliases.jsonl and make records GitHub-friendly."""
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                "code": row["code"],
                "system": row["system"],
                "type": row["type"],
                "name": row["name"],
                "ttys": row.get("ttys", []),
                "priority": row.get("priority", 100),
                "archive": row.get("archive", False),
            }
        )
    return compact


def build_alias_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_rows:
        concept_type = str(row["type"])
        code = str(row["code"])
        priority = int(row.get("priority", 100))
        for alias in row.get("aliases") or []:
            key = normalize_key(str(alias))
            if not key:
                continue
            item = aliases.setdefault(
                (concept_type, key),
                {
                    "alias_norm": key,
                    "type": concept_type,
                    "candidates": [],
                },
            )
            item["candidates"].append({"code": code, "priority": priority})

    output: list[dict[str, Any]] = []
    for item in aliases.values():
        dedup: dict[str, int] = {}
        for cand in item["candidates"]:
            code = str(cand["code"])
            dedup[code] = min(dedup.get(code, 999), int(cand["priority"]))
        item["candidates"] = [
            {"code": code, "priority": priority}
            for code, priority in sorted(dedup.items(), key=lambda pair: (pair[1], pair[0]))[:20]
        ]
        output.append(item)
    return sorted(output, key=lambda row: (row["type"], row["alias_norm"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build slim ICD-10/RxNorm candidate artifacts.")
    parser.add_argument("--icd10", type=Path, default=ROOT / "data" / "icd10" / "icd10.jsonl")
    parser.add_argument("--rxnconso", type=Path, default=ROOT / "data" / "rxnorm" / "RXNCONSO.RRF")
    parser.add_argument("--rxnarchive", type=Path, default=ROOT / "data" / "rxnorm" / "RXNATOMARCHIVE.RRF")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "candidates")
    parser.add_argument(
        "--existing-icd-candidates",
        type=Path,
        default=None,
        help="Reuse an already-built icd10_candidates.jsonl when raw ICD-10 jsonl is not present.",
    )
    args = parser.parse_args(argv)

    icd_rows = build_icd10_rows(args.icd10)
    if not icd_rows and args.existing_icd_candidates is not None:
        icd_rows = read_candidate_rows(args.existing_icd_candidates)
    rx_rows = build_rxnorm_rows(args.rxnconso, args.rxnarchive)
    all_rows = icd_rows + rx_rows
    alias_rows = build_alias_rows(all_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "icd10_candidates.jsonl", icd_rows)
    write_jsonl(args.out_dir / "rxnorm_candidates.jsonl", compact_rxnorm_rows(rx_rows))
    write_jsonl(args.out_dir / "candidate_aliases.jsonl", alias_rows)

    manifest = {
        "version": 1,
        "purpose": "Slim runtime candidate data for ICD-10 diagnosis and RxNorm drug mapping.",
        "sources": {
            "icd10": str(args.icd10),
            "rxnconso": str(args.rxnconso),
            "rxnarchive": str(args.rxnarchive),
        },
        "counts": {
            "icd10_candidates": len(icd_rows),
            "rxnorm_candidates": len(rx_rows),
            "aliases": len(alias_rows),
        },
        "artifacts": [
            "icd10_candidates.jsonl",
            "rxnorm_candidates.jsonl",
            "candidate_aliases.jsonl",
            "candidate_manifest.json",
        ],
    }
    (args.out_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
