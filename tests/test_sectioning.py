from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extraction.sectioning import detect_sections, split_chunks


class SectioningTest(unittest.TestCase):
    def test_detects_vietnamese_sections(self) -> None:
        text = (
            "1. Tiền sử bệnh nội khoa\n"
            "- Tăng huyết áp\n\n"
            "Kết quả xét nghiệm:\n"
            "WBC:14,43; NEUT%:76,4;"
        )
        sections = detect_sections(text)
        self.assertEqual([section.name for section in sections], ["history", "labs"])

    def test_chunks_keep_exact_source_offsets(self) -> None:
        text = "Kết quả xét nghiệm:\n" + ("WBC:14,43; " * 220)
        chunks = split_chunks(text, max_chars=360, overlap=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(text[chunk.start : chunk.end], chunk.text)
            self.assertLess(chunk.start, chunk.end)


if __name__ == "__main__":
    unittest.main()
