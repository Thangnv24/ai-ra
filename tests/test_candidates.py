from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import SYSTEM_ICD10, SYSTEM_RXNORM, TYPE_DIAGNOSIS, TYPE_DRUG
from knowledge.candidates import load_slim_candidate_index


class SlimCandidateIndexTest(unittest.TestCase):
    def test_builds_aliases_from_candidate_rows_when_alias_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp)
            row = {
                "code": "C90.0",
                "system": SYSTEM_ICD10,
                "type": TYPE_DIAGNOSIS,
                "name_vi": "Đa u tủy xương",
                "alias_norms": ["da u tuy xuong"],
                "priority": 10,
            }
            (candidate_dir / "icd10_candidates.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            index = load_slim_candidate_index(candidate_dir)
            hits = index.lookup("Đa u tủy xương", TYPE_DIAGNOSIS)

        self.assertEqual([hit.record.code for hit in hits], ["C90.0"])

    def test_drug_query_normalizes_attached_strength_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp)
            row = {
                "code": "317300",
                "system": SYSTEM_RXNORM,
                "type": TYPE_DRUG,
                "name": "aspirin 325 MG Oral Tablet",
                "priority": 2,
            }
            alias = {
                "alias_norm": "aspirin 325 mg",
                "type": TYPE_DRUG,
                "candidates": [{"code": "317300", "priority": 2}],
            }
            (candidate_dir / "rxnorm_candidates.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (candidate_dir / "candidate_aliases.jsonl").write_text(
                json.dumps(alias, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            index = load_slim_candidate_index(candidate_dir)
            hits = index.lookup("aspirin 325mg x 1", TYPE_DRUG)

        self.assertEqual([hit.record.code for hit in hits], ["317300"])


if __name__ == "__main__":
    unittest.main()
