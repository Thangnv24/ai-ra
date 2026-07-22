"""Tests for the layered offline evaluator."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from core.config import TYPE_DIAGNOSIS
from knowledge.candidates import CandidateHit, CandidateRecord
from scripts.analyze_candidate_mapping import _classify_outcome
from scripts.build_candidate_emission_policy import build_policy
from scripts.evaluate_layers import SetScores, SpanCounts, evaluate_layers


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

    def test_candidate_diagnostics_separate_retrieval_and_selector_failures(self) -> None:
        record = CandidateRecord("I10", "Essential hypertension", "ICD10", TYPE_DIAGNOSIS)
        records = {(TYPE_DIAGNOSIS, "I10"): record}
        hit = CandidateHit(record, "exact", 0.95)

        self.assertEqual(
            _classify_outcome(
                ("I10",), (), [hit], eligible=True, records=records, concept_type=TYPE_DIAGNOSIS
            ),
            "selector_abstain",
        )
        self.assertEqual(
            _classify_outcome(
                ("I10",), (), [], eligible=True, records=records, concept_type=TYPE_DIAGNOSIS
            ),
            "retrieval_miss",
        )


class EvaluateLayersTests(unittest.TestCase):
    def test_candidate_policy_builder_requires_cross_file_support(self) -> None:
        repeated = {
            "text": "duloxetine",
            "type": "THU\u1ed0C",
            "line_profile": "long_line",
            "expected": ["72625"],
            "predicted": ["476253"],
        }
        unique = {
            "file": "3.json",
            "text": "single-use",
            "type": "THU\u1ed0C",
            "line_profile": "short_line",
            "expected": ["1"],
            "predicted": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "baseline.json"
            report_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {**repeated, "file": "1.json"},
                            {**repeated, "file": "2.json"},
                            unique,
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            policy = build_policy(
                baseline_report=report_path,
                min_file_support=2,
                min_weighted_gain=1.0,
            )

        self.assertEqual(policy["summary"]["rules"], 1)
        self.assertEqual(policy["rules"][0]["candidates"], ["72625"])

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


if __name__ == "__main__":
    unittest.main()
