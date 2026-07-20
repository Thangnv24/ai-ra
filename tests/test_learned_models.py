"""Tests for dependency-free learned extraction artifacts."""

from __future__ import annotations

import unittest

from core.config import TYPE_SYMPTOM
from extraction.assertion_model import AssertionClassifier
from extraction.context import ContextDetector
from extraction.learned_models import SpanAcceptanceModel, TokenSpanModel
from extraction.ner import SpanCandidate


class LearnedModelTests(unittest.TestCase):
    def test_assertion_classifier_can_recover_long_narrative_cue(self) -> None:
        classifier = AssertionClassifier(
            {
                "isNegated": {
                    "bias": -1.0,
                    "weights": {"has_negation_cue": 3.0},
                    "threshold": 0.5,
                }
            }
        )
        mention = "khó thở"
        text = "Bệnh nhân " + ("có diễn biến kéo dài, " * 18) + f"phủ nhận {mention}."
        start = text.index(mention)

        assertions = ContextDetector(classifier).assertions_for(
            text, start, start + len(mention), TYPE_SYMPTOM
        )

        self.assertIn("isNegated", assertions)

    def test_token_model_emits_exact_source_span(self) -> None:
        model = TokenSpanModel(
            labels=("O", f"B:{TYPE_SYMPTOM}", f"I:{TYPE_SYMPTOM}"),
            weights={
                "token=ho": {f"B:{TYPE_SYMPTOM}": 3.0},
                "bias": {"O": 0.1},
            },
        )

        spans = model.propose("Bệnh nhân ho")

        self.assertEqual([(span.text, span.type) for span in spans], [("ho", TYPE_SYMPTOM)])

    def test_acceptance_model_returns_probability(self) -> None:
        model = SpanAcceptanceModel(
            bias=-1.0,
            weights={f"type={TYPE_SYMPTOM}": 2.0},
            thresholds={TYPE_SYMPTOM: 0.6},
        )
        span = SpanCandidate(0, 2, "ho", TYPE_SYMPTOM, 0.7, "llm")

        score = model.score("ho", span, "document", "document", None)

        self.assertIsNotNone(score)
        self.assertGreater(score, 0.5)
        self.assertEqual(model.threshold_for(TYPE_SYMPTOM, "llm"), 0.6)


if __name__ == "__main__":
    unittest.main()
