"""Tests for cross-suite promotion guards."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("promotion_gate", ROOT / "scripts" / "promotion_gate.py")
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


class PromotionGateTests(unittest.TestCase):
    def test_identical_reviewed_run_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_dir, gold_dir, baseline_dir, prediction_dir = self._suite(Path(temp))

            baseline = GATE.evaluate_folder(input_dir, gold_dir, baseline_dir)
            prediction = GATE.evaluate_folder(input_dir, gold_dir, prediction_dir)
            result = GATE.check_promotion(prediction, baseline)

            self.assertTrue(result.passed, result.failures)

    def test_candidate_regression_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_dir, gold_dir, baseline_dir, prediction_dir = self._suite(Path(temp))
            wrong = self._concept("I11.9")
            (prediction_dir / "1.json").write_text(json.dumps([wrong]), encoding="utf-8")

            baseline = GATE.evaluate_folder(input_dir, gold_dir, baseline_dir)
            prediction = GATE.evaluate_folder(input_dir, gold_dir, prediction_dir)
            result = GATE.check_promotion(prediction, baseline)

            self.assertFalse(result.passed)
            self.assertTrue(any("candidate_score regressed" in failure for failure in result.failures))

    def test_missing_or_mixed_files_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_dir, gold_dir, baseline_dir, prediction_dir = self._suite(Path(temp))
            (prediction_dir / "1.json").unlink()
            (prediction_dir / "part2.json").write_text("[]", encoding="utf-8")

            prediction = GATE.evaluate_folder(input_dir, gold_dir, prediction_dir)
            result = GATE.check_promotion(prediction, None)

            self.assertFalse(result.passed)
            self.assertTrue(any("missing prediction files" in failure for failure in result.failures))
            self.assertTrue(any("unexpected prediction files" in failure for failure in result.failures))

    def test_invalid_baseline_is_not_accepted_as_a_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_dir, gold_dir, baseline_dir, prediction_dir = self._suite(Path(temp))
            (baseline_dir / "1.json").write_text("[]", encoding="utf-8")
            (baseline_dir / "_INCOMPLETE_RUN.txt").write_text("failed", encoding="utf-8")

            baseline = GATE.evaluate_folder(input_dir, gold_dir, baseline_dir)
            prediction = GATE.evaluate_folder(input_dir, gold_dir, prediction_dir)
            result = GATE.check_promotion(prediction, baseline)

            self.assertFalse(result.passed)
            self.assertIn("baseline folder has _INCOMPLETE_RUN.txt", result.failures)

    def _suite(self, root: Path) -> tuple[Path, Path, Path, Path]:
        input_dir = root / "input"
        gold_dir = root / "gold"
        baseline_dir = root / "baseline"
        prediction_dir = root / "prediction"
        for path in (input_dir, gold_dir, baseline_dir, prediction_dir):
            path.mkdir()
        (input_dir / "1.txt").write_text("tang huyet ap", encoding="utf-8")
        payload = [self._concept("I10")]
        for path in (gold_dir, baseline_dir, prediction_dir):
            (path / "1.json").write_text(json.dumps(payload), encoding="utf-8")
        return input_dir, gold_dir, baseline_dir, prediction_dir

    @staticmethod
    def _concept(code: str) -> dict[str, object]:
        return {
            "text": "tang huyet ap",
            "type": "CH\u1ea8N_\u0110O\u00c1N",
            "assertions": [],
            "position": [0, 13],
            "candidates": [code],
        }


if __name__ == "__main__":
    unittest.main()
