from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_TYPES, TYPE_DIAGNOSIS
from integrations.prompts import build_decision_prompt, build_entity_extraction_prompt


class PromptTemplateTest(unittest.TestCase):
    def test_decision_prompt_uses_configured_type_labels(self) -> None:
        payload = json.loads(build_decision_prompt("text", []))
        self.assertEqual(payload["allowed_types"], list(ALLOWED_TYPES))
        self.assertEqual(payload["response_schema"]["decisions"][0]["final_type"], TYPE_DIAGNOSIS)

    def test_entity_prompt_uses_configured_type_labels(self) -> None:
        payload = json.loads(
            build_entity_extraction_prompt(
                {"chunk_id": "c1", "section": "document", "text": "Bệnh nhân ho."}
            )
        )
        self.assertEqual(payload["allowed_types"], list(ALLOWED_TYPES))


if __name__ == "__main__":
    unittest.main()
