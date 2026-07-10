from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import SYSTEM_ICD10, SYSTEM_RXNORM, TYPE_DIAGNOSIS, TYPE_DRUG, get_paths
from knowledge.ontology import seed_entries


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_concepts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in seed_entries():
        rows.append(
            {
                "concept_id": f"{entry.system}:{entry.code}",
                "code": entry.code,
                "canonical_name": entry.name,
                "type": entry.concept_type,
                "source_system": entry.system,
                "aliases": list(entry.aliases),
                "source": "manual_seed",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Build merged local medical knowledge base JSONL files")
    parser.parse_args(argv)

    concepts = seed_concepts()
    for row in read_jsonl(paths.data_processed / "icd10" / "icd10_concepts.jsonl"):
        code = str(row.get("code") or "")
        if code:
            concepts.append(
                {
                    "concept_id": f"{SYSTEM_ICD10}:{code}",
                    "code": code,
                    "canonical_name": row.get("canonical_name") or "",
                    "type": TYPE_DIAGNOSIS,
                    "source_system": SYSTEM_ICD10,
                    "aliases": row.get("aliases") or [],
                    "source": row.get("source") or "icd10",
                }
            )
    for row in read_jsonl(paths.data_processed / "rxnorm" / "rxnorm_concepts.jsonl"):
        code = str(row.get("rxcui") or "")
        if code:
            concepts.append(
                {
                    "concept_id": f"{SYSTEM_RXNORM}:{code}",
                    "code": code,
                    "canonical_name": row.get("canonical_name") or "",
                    "type": TYPE_DRUG,
                    "source_system": SYSTEM_RXNORM,
                    "aliases": row.get("aliases") or [],
                    "source": row.get("source") or "rxnorm",
                }
            )

    dedup: dict[str, dict[str, object]] = {}
    for concept in concepts:
        cid = str(concept["concept_id"])
        if cid not in dedup:
            dedup[cid] = concept
        else:
            aliases = set(dedup[cid].get("aliases") or [])
            aliases.update(concept.get("aliases") or [])
            dedup[cid]["aliases"] = sorted(a for a in aliases if a)
    concepts = list(dedup.values())

    aliases = read_jsonl(paths.data_processed / "vi_aliases" / "aliases.jsonl")
    for concept in concepts:
        for alias in [concept.get("canonical_name"), *(concept.get("aliases") or [])]:
            if alias:
                aliases.append(
                    {
                        "alias": alias,
                        "canonical_name": concept.get("canonical_name"),
                        "type": concept.get("type"),
                        "candidate_codes": [concept.get("code")] if concept.get("type") in {TYPE_DIAGNOSIS, TYPE_DRUG} else [],
                        "source": concept.get("source"),
                    }
                )

    out = paths.data_processed
    write_jsonl(out / "concepts.jsonl", concepts)
    write_jsonl(out / "aliases.jsonl", aliases)
    write_jsonl(out / "drug_aliases.jsonl", [r for r in aliases if r.get("type") == TYPE_DRUG])
    write_jsonl(out / "disease_aliases.jsonl", [r for r in aliases if r.get("type") == TYPE_DIAGNOSIS])
    write_jsonl(out / "lab_aliases.jsonl", [r for r in aliases if r.get("type") == "TÊN_XÉT_NGHIỆM"])
    write_jsonl(out / "symptom_aliases.jsonl", [r for r in aliases if r.get("type") == "TRIỆU_CHỨNG"])
    print(json.dumps({"concepts": len(concepts), "aliases": len(aliases), "output_dir": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
