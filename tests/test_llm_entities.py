from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import TYPE_SYMPTOM
from extraction.llm_entities import LLMEntityExtractor, align_quote_in_chunk
from extraction.sectioning import TextChunk


@dataclass(frozen=True)
class FakeResult:
    ok: bool
    data: dict
    error: str | None = None


class FakeLLMClient:
    enabled = True

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> FakeResult:
        return FakeResult(
            ok=True,
            data={
                "mentions": [
                    {"quote": "ho đờm xanh", "type": TYPE_SYMPTOM, "confidence": 0.91},
                    {"quote": "không có trong text", "type": TYPE_SYMPTOM, "confidence": 0.99},
                ]
            },
        )


class LLMEntityExtractionTest(unittest.TestCase):
    def test_align_quote_uses_global_offsets(self) -> None:
        text = "Bệnh nhân ho đờm xanh."
        chunk = TextChunk("c1", "document", 0, len(text), text)
        self.assertEqual(align_quote_in_chunk(chunk, "ho đờm xanh"), (10, 21))

    def test_fake_llm_mentions_are_aligned_and_filtered(self) -> None:
        text = "Bệnh nhân ho đờm xanh. Không sốt."
        spans, summary = LLMEntityExtractor(FakeLLMClient()).extract(text)
        self.assertEqual(summary.chunks, 1)
        self.assertEqual(summary.mentions, 2)
        self.assertEqual(summary.aligned, 1)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "ho đờm xanh")
        self.assertEqual(text[spans[0].start : spans[0].end], spans[0].text)


if __name__ == "__main__":
    unittest.main()
