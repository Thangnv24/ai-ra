from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import SYSTEM_ICD10, SYSTEM_RXNORM, TYPE_DIAGNOSIS, TYPE_DRUG, get_paths
from medkg.normalization import normalize_key
from medkg.ontology import OntologyEntry, OntologyIndex, seed_entries


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


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Build ontology and alias indexes from processed KB")
    parser.parse_args(argv)

    entries = seed_entries()
    concepts = read_jsonl(paths.data_processed / "concepts.jsonl")
    for row in concepts:
        source_system = str(row.get("source_system") or "")
        concept_type = str(row.get("type") or "")
        code = str(row.get("code") or "")
        name = str(row.get("canonical_name") or "")
        if source_system in {SYSTEM_ICD10, SYSTEM_RXNORM} and concept_type in {TYPE_DIAGNOSIS, TYPE_DRUG} and code and name:
            entries.append(OntologyEntry(code, name, source_system, concept_type, tuple(str(a) for a in row.get("aliases") or []), 50))
    index = OntologyIndex(entries)
    paths.data_indexes.mkdir(parents=True, exist_ok=True)
    (paths.data_indexes / "ontology_index.json").write_text(
        json.dumps({"version": 2, "entries": index.to_json_data()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aliases = read_jsonl(paths.data_processed / "aliases.jsonl")
    alias_exact: dict[str, list[dict[str, object]]] = {}
    alias_norm: dict[str, list[dict[str, object]]] = {}
    for row in aliases:
        alias = str(row.get("alias") or "")
        if not alias:
            continue
        compact = {
            "canonical_name": row.get("canonical_name"),
            "type": row.get("type"),
            "candidate_codes": row.get("candidate_codes") or [],
            "source": row.get("source"),
        }
        alias_exact.setdefault(alias, []).append(compact)
        alias_norm.setdefault(normalize_key(alias), []).append(compact)
    (paths.data_indexes / "alias_exact.json").write_text(json.dumps(alias_exact, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths.data_indexes / "alias_norm.json").write_text(json.dumps(alias_norm, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "concept_count": len(index.entries),
        "alias_exact_count": len(alias_exact),
        "alias_norm_count": len(alias_norm),
        "artifacts": ["ontology_index.json", "alias_exact.json", "alias_norm.json", "kb_manifest.json"],
    }
    (paths.data_indexes / "kb_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
