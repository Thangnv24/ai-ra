"""Tests for the layered offline evaluator."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.evaluate_layers import SetScores, SpanCounts, _error_taxonomy, evaluate_layers


class MetricAccumulatorTests(unittest.TestCase):
    def test_span_counts_preserve_duplicate_occurrences(self) -> None:
        gold = Counter({(0, 4, "TYPE"): 2})
        predicted = Counter({(0, 4, "TYPE"): 1, (8, 12, "TYPE"): 1})
        counts = SpanCounts()

        counts.add(gold, predicted)

        self.assertEqual(
            (counts.true_positive, counts.false_positive, counts.false_negative),
            (1, 1, 1),
        )

    def test_candidate_scores_use_gold_list_weight(self) -> None:
        scores = SetScores()
        scores.add(["A"], ["A"], weighted=True)
        scores.add([], ["B"], weighted=True)

        summary = scores.to_dict()

        self.assertEqual(summary["mean_jaccard"], 0.5)
        self.assertEqual(summary["weighted_jaccard"], round(2 / 3, 6))

    def test_error_taxonomy_separates_type_and_boundary_errors(self) -> None:
        gold = Counter({(10, 20, "A"): 1, (30, 40, "B"): 1})
        predicted = Counter({(10, 20, "B"): 1, (32, 40, "B"): 1})

        errors = _error_taxonomy(gold, predicted)

        self.assertEqual(errors["wrong_type"], 1)
        self.assertEqual(errors["boundary_too_short"], 1)


class EvaluateLayersTests(unittest.TestCase):
    def test_evaluates_oracle_layers_and_exact_prediction_folder(self) -> None:
        text = "Triệu chứng: đánh trống ngực."
        mention = "đánh trống ngực"
        start = text.index(mention)
        concepts = [
            {
                "text": mention,
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "position": [start, start + len(mention)],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            gold_dir = root / "gold"
            prediction_dir = root / "prediction"
            input_dir.mkdir()
            gold_dir.mkdir()
            prediction_dir.mkdir()
            (input_dir / "1.txt").write_text(text, encoding="utf-8")
            payload = json.dumps(concepts, ensure_ascii=False)
            (gold_dir / "1.json").write_text(payload, encoding="utf-8")
            (prediction_dir / "1.json").write_text(payload, encoding="utf-8")

            report = evaluate_layers(
                input_dir=input_dir,
                gold_dir=gold_dir,
                prediction_dir=prediction_dir,
            )

        self.assertEqual(report["files"], 1)
        self.assertEqual(report["validation"]["gold_offset_mismatches"], 0)
        self.assertEqual(report["assertion_oracle_span"]["exact_set_rate"], 1.0)
        self.assertEqual(report["end_to_end"]["final_score_estimate"], 100.0)
        self.assertEqual(
            report["prediction_diagnostics"]["exact_span_and_type"]["f1"],
            1.0,
        )
        self.assertEqual(
            report["prediction_diagnostics"]["assertion_on_exact_spans"]["exact_set_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
