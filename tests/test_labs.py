from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import TYPE_TEST_NAME, TYPE_TEST_RESULT
from extraction.labs import extract_lab_spans


class LabExtractionTest(unittest.TestCase):
    def test_extracts_abbreviation_and_parenthetical_lab_pairs(self) -> None:
        text = "WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4; LYPH%:12,8;"
        spans = extract_lab_spans(text)
        items = [(span.text, span.type, (span.start, span.end)) for span in spans]

        self.assertIn(("WBC", TYPE_TEST_NAME, (0, 3)), items)
        self.assertIn(("14,43", TYPE_TEST_RESULT, (4, 9)), items)
        self.assertIn(("NEUT% (Tỷ lệ % bạch cầu trung tính)", TYPE_TEST_NAME, (11, 46)), items)
        self.assertIn(("76,4", TYPE_TEST_RESULT, (47, 51)), items)

    def test_extracts_vietnamese_lab_name_with_la_separator(self) -> None:
        text = "bilirubin toàn phần (tbili) là 2.4"
        spans = extract_lab_spans(text)
        pairs = [(span.text, span.type) for span in spans]
        self.assertEqual(
            pairs,
            [
                ("bilirubin toàn phần (tbili)", TYPE_TEST_NAME),
                ("2.4", TYPE_TEST_RESULT),
            ],
        )
        for span in spans:
            self.assertEqual(text[span.start : span.end], span.text)

    def test_lab_name_drops_bullet_prefix(self) -> None:
        text = "- ast (aspartate aminotransferase) là 319"
        spans = extract_lab_spans(text)
        self.assertEqual(spans[0].text, "ast (aspartate aminotransferase)")
        self.assertEqual(text[spans[0].start : spans[0].end], spans[0].text)


if __name__ == "__main__":
    unittest.main()
