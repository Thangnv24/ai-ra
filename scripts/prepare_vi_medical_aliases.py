from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medkg.config import TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_NAME, get_paths


ALIASES: list[dict[str, object]] = [
    {"alias": "bệnh trào ngược dạ dày thực quản", "canonical_name": "Gastro-esophageal reflux disease", "type": TYPE_DIAGNOSIS, "candidate_codes": ["K21.0", "K21.9"], "source": "statement"},
    {"alias": "trào ngược dạ dày thực quản", "canonical_name": "Gastro-esophageal reflux disease", "type": TYPE_DIAGNOSIS, "candidate_codes": ["K21.0", "K21.9"], "source": "statement"},
    {"alias": "GERD", "canonical_name": "Gastro-esophageal reflux disease", "type": TYPE_DIAGNOSIS, "candidate_codes": ["K21.9"], "source": "manual_seed"},
    {"alias": "tăng huyết áp", "canonical_name": "Essential hypertension", "type": TYPE_DIAGNOSIS, "candidate_codes": ["I10"], "source": "sample"},
    {"alias": "tăng lipid máu", "canonical_name": "Hyperlipidemia", "type": TYPE_DIAGNOSIS, "candidate_codes": ["E78.5"], "source": "sample"},
    {"alias": "bệnh tim mạch do xơ vữa động mạch", "canonical_name": "Atherosclerotic cardiovascular disease", "type": TYPE_DIAGNOSIS, "candidate_codes": ["I25.10"], "source": "sample"},
    {"alias": "bệnh trào ngược dạ dày-thực quản không có viêm thực quản", "canonical_name": "GERD without esophagitis", "type": TYPE_DIAGNOSIS, "candidate_codes": ["K21.9"], "source": "sample"},
    {"alias": "tắc nghẽn đường mật", "canonical_name": "Obstruction of bile duct", "type": TYPE_DIAGNOSIS, "candidate_codes": ["K83.1"], "source": "sample"},
    {"alias": "giãn đường mật", "canonical_name": "Obstruction of bile duct", "type": TYPE_DIAGNOSIS, "candidate_codes": ["K83.1"], "source": "sample"},
    {"alias": "nốt tuyến giáp", "canonical_name": "Thyroid nodule", "type": TYPE_DIAGNOSIS, "candidate_codes": ["E04.1"], "source": "manual_seed"},
    {"alias": "nốt tuyến giáp thùy trái", "canonical_name": "Thyroid nodule", "type": TYPE_DIAGNOSIS, "candidate_codes": ["E04.1"], "source": "manual_seed"},
    {"alias": "tổn thương tuyến giáp", "canonical_name": "Disorder of thyroid", "type": TYPE_DIAGNOSIS, "candidate_codes": ["E07.9"], "source": "manual_seed"},
    {"alias": "kết quả chọc hút bất thường", "canonical_name": "Abnormal cytology finding", "type": TYPE_DIAGNOSIS, "candidate_codes": ["R89.6"], "source": "manual_seed"},
    {"alias": "bất thường tế bào học", "canonical_name": "Abnormal cytology finding", "type": TYPE_DIAGNOSIS, "candidate_codes": ["R89.6"], "source": "manual_seed"},
    {"alias": "đau bụng", "canonical_name": "abdominal pain", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "đau bụng vùng thượng vị", "canonical_name": "epigastric pain", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "buồn nôn", "canonical_name": "nausea", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "sốt", "canonical_name": "fever", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "ớn lạnh", "canonical_name": "chills", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "nôn", "canonical_name": "vomiting", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "táo bón", "canonical_name": "constipation", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "ho", "canonical_name": "cough", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "tiểu khó", "canonical_name": "dysuria", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "hạ huyết áp", "canonical_name": "hypotension", "type": TYPE_DIAGNOSIS, "candidate_codes": ["I95.9"], "source": "sample"},
    {"alias": "khó nuốt", "canonical_name": "dysphagia", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "manual_seed"},
    {"alias": "khó thở", "canonical_name": "dyspnea", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "sample"},
    {"alias": "khàn tiếng", "canonical_name": "hoarseness", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "manual_seed"},
    {"alias": "đau nhức", "canonical_name": "pain", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "statement"},
    {"alias": "lo âu", "canonical_name": "anxiety", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "statement"},
    {"alias": "mất ngủ", "canonical_name": "insomnia", "type": TYPE_SYMPTOM, "candidate_codes": [], "source": "statement"},
]

LABS = ["WBC", "bạch cầu", "AST", "aspartate aminotransferase", "ALT", "alanine aminotransferase", "phosphatase kiềm", "alkaline phosphatase", "AP", "bilirubin toàn phần", "TBili", "NEUT%", "LYPH%", "tổng phân tích tế bào máu", "xét nghiệm tế bào học", "cấy máu"]
DRUGS = ["levophed", "norepinephrine", "propofol", "phentolamine", "aspirin", "amlodipine", "metoprolol", "chlorpheniramine", "capsaicin", "acetaminophen", "clonazepam", "nystatin", "pravastatin", "senna", "docusate sodium", "guaifenesin"]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            alias = (row.get("term") or "").strip()
            concept_type = (row.get("type") or "").strip()
            if alias and concept_type:
                rows.append({"alias": alias, "canonical_name": alias, "type": concept_type, "candidate_codes": [], "source": row.get("source") or "local_lexicon"})
    return rows


def main(argv: list[str] | None = None) -> int:
    paths = get_paths(ROOT)
    parser = argparse.ArgumentParser(description="Prepare Vietnamese clinical alias JSONL resources")
    parser.parse_args(argv)

    all_rows = ALIASES + load_existing_csv(paths.data_external / "vietnamese_clinical_lexicon.csv")
    all_rows.extend({"alias": lab, "canonical_name": lab, "type": TYPE_TEST_NAME, "candidate_codes": [], "source": "manual_seed"} for lab in LABS)
    all_rows.extend({"alias": drug, "canonical_name": drug, "type": TYPE_DRUG, "candidate_codes": [], "source": "manual_seed"} for drug in DRUGS)

    dedup: dict[tuple[str, str], dict[str, object]] = {}
    for row in all_rows:
        key = (str(row["alias"]).casefold(), str(row["type"]))
        dedup.setdefault(key, row)
    rows = list(dedup.values())

    out_dir = paths.data_processed / "vi_aliases"
    write_jsonl(out_dir / "aliases.jsonl", rows)
    write_jsonl(out_dir / "disease_aliases.jsonl", [r for r in rows if r["type"] == TYPE_DIAGNOSIS])
    write_jsonl(out_dir / "drug_aliases.jsonl", [r for r in rows if r["type"] == TYPE_DRUG])
    write_jsonl(out_dir / "symptom_aliases.jsonl", [r for r in rows if r["type"] == TYPE_SYMPTOM])
    write_jsonl(out_dir / "lab_aliases.jsonl", [r for r in rows if r["type"] == TYPE_TEST_NAME])
    print(json.dumps({"output_dir": str(out_dir), "aliases": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
