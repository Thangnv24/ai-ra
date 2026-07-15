"""End-to-end LLM-required inference pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from core.config import (
    ALLOWED_ASSERTIONS,
    ALLOWED_TYPES,
    ASSERTION_TYPES,
    CODED_TYPES,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
    get_paths,
    get_settings,
)
from core.io import discover_input_files, output_path_for, read_text, write_output
from core.schema import Concept, validate_output
from extraction.context import ContextDetector
from extraction.llm_entities import LLMEntityExtractor
from extraction.ner import MedicalNER, SpanCandidate
from integrations.openai_client import ApiLLMClient
from integrations.prompts import SYSTEM_PROMPT, build_decision_prompt
from knowledge.candidates import load_slim_candidate_index
from knowledge.ontology import OntologyIndex, load_ontology_index
from knowledge.reasoning import infer_relations
from knowledge.retrieval import CandidateRetriever
from services.postprocess import refine_concepts


@dataclass(frozen=True)
class FileTiming:
    input_path: str
    output_path: str
    concepts: int
    seconds: float


@dataclass(frozen=True)
class RunSummary:
    files: int
    concepts: int
    total_seconds: float
    average_seconds: float
    p50_seconds: float
    p95_seconds: float
    timings: tuple[FileTiming, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "files": self.files,
            "concepts": self.concepts,
            "total_seconds": round(self.total_seconds, 6),
            "average_seconds": round(self.average_seconds, 6),
            "p50_seconds": round(self.p50_seconds, 6),
            "p95_seconds": round(self.p95_seconds, 6),
            "timings": [
                {
                    "input_path": item.input_path,
                    "output_path": item.output_path,
                    "concepts": item.concepts,
                    "seconds": round(item.seconds, 6),
                }
                for item in self.timings
            ],
        }


class MedicalKGPipeline:
    def __init__(self, index: OntologyIndex | None = None, root: Path | None = None):
        self.root = root
        paths = get_paths(root)
        self.settings = get_settings(root)
        self._ensure_indexes(paths)
        self.index = index or load_ontology_index(paths.index_file, paths.data_raw, paths.data_external)
        self.slim_candidate_index = load_slim_candidate_index(paths.root / "data" / "candidates")
        self.ner = MedicalNER((paths.data_external / "vietnamese_clinical_lexicon.csv",))
        self.context = ContextDetector()
        self.retriever = CandidateRetriever(self.index, self.slim_candidate_index)
        self.llm = ApiLLMClient(self.settings)
        self.llm_entity_extractor = LLMEntityExtractor(self.llm)

    def _ensure_indexes(self, paths) -> None:
        if paths.index_file.exists():
            return
        build_script = paths.root / "scripts" / "build_indexes.py"
        if not build_script.exists() or not (paths.data_processed / "concepts.jsonl").exists():
            return
        try:
            subprocess.run([sys.executable, str(build_script)], cwd=paths.root, check=False, capture_output=True, text=True, timeout=60)
        except Exception:
            return

    def process_text(self, text: str) -> list[Concept]:
        concepts, _ = self.process_text_with_meta(text)
        return concepts

    def process_text_with_meta(self, text: str, mode: str | None = None) -> tuple[list[Concept], dict[str, object]]:
        if mode is not None and mode != "llm_full_doc":
            raise ValueError("Only llm_full_doc mode is supported; local fallback modes were removed")
        requested_mode = "llm_full_doc"
        if not self.settings.llm_enabled:
            raise RuntimeError("LLM is required but disabled")

        spans = self.ner.extract(text)
        llm_spans, summary = self.llm_entity_extractor.extract(text)
        llm_entity_meta = summary.to_dict()
        spans = _merge_span_candidates([*spans, *llm_spans])

        concepts: list[Concept] = []
        for span in spans:
            assertions = self.context.assertions_for(text, span.start, span.end, span.type)
            candidates = self.retriever.candidates_for(span.text, span.type)
            concepts.append(
                Concept(
                    text=span.text,
                    type=span.type,
                    position=(span.start, span.end),
                    assertions=assertions,
                    candidates=candidates,
                )
            )
        concepts = sorted(concepts, key=lambda c: (c.position[0], c.position[1], c.type))
        meta: dict[str, object] = {
            "mode": requested_mode,
            "mode_used": "llm_full_doc",
            "llm_required": True,
            "llm_used": False,
            "llm_error": None,
            "llm_entity": llm_entity_meta,
        }
        concepts, meta = self._apply_llm_decisions(text, concepts, meta)
        before_postprocess = len(concepts)
        concepts = refine_concepts(text, concepts, retriever=self.retriever, context_detector=self.context)
        meta["postprocess"] = {
            "input_concepts": before_postprocess,
            "output_concepts": len(concepts),
        }
        # Keep relation inference available without changing the public schema.
        infer_relations(concepts)
        errors = validate_output([concept.to_dict() for concept in concepts], source_text=text)
        if errors:
            raise ValueError("pipeline generated invalid output: " + "; ".join(errors[:5]))
        return concepts, meta

    def process_file(self, input_path: Path) -> list[Concept]:
        return self.process_text(read_text(input_path))

    def _apply_llm_decisions(
        self,
        text: str,
        concepts: list[Concept],
        meta: dict[str, object],
    ) -> tuple[list[Concept], dict[str, object]]:
        if not self.settings.llm_enabled:
            raise RuntimeError("LLM decision pass is required but disabled")

        mention_payload: list[dict[str, object]] = []
        decision_concepts: dict[str, Concept] = {}
        passthrough: list[Concept] = []
        passthrough_by_type: dict[str, int] = {}
        retrieved_by_id: dict[str, set[str]] = {}
        for idx, concept in enumerate(concepts):
            candidate_rows = (
                _compact_candidate_rows(self.retriever.candidate_rows_for(concept.text, concept.type, limit=10))
                if concept.type == TYPE_DIAGNOSIS
                else []
            )
            if not _needs_llm_decision(concept, candidate_rows):
                passthrough.append(concept)
                passthrough_by_type[concept.type] = passthrough_by_type.get(concept.type, 0) + 1
                continue

            mention_id = f"m{idx + 1}"
            retrieved_by_id[mention_id] = {row["code"] for row in candidate_rows}
            decision_concepts[mention_id] = concept
            start, end = concept.position
            mention_payload.append(
                {
                    "mention_id": mention_id,
                    "text": concept.text,
                    "position": [start, end],
                    "proposed_type": concept.type,
                    "local_context": text[max(0, start - 120) : min(len(text), end + 120)],
                    "rule_assertions": list(concept.assertions),
                    "retrieved_candidates": candidate_rows,
                }
            )

        if not mention_payload:
            meta["mode_used"] = "llm_full_doc"
            meta["llm_used"] = True
            meta["llm_decisions"] = 0
            meta["llm_decision_scope"] = "ambiguous_diagnosis_and_labs"
            meta["llm_decision_passthrough"] = len(passthrough)
            meta["llm_decision_passthrough_by_type"] = passthrough_by_type
            return sorted(passthrough, key=lambda c: (c.position[0], c.position[1], c.type)), meta

        result = self.llm.chat_json(SYSTEM_PROMPT, build_decision_prompt("", mention_payload))
        if not result.ok or not isinstance(result.data, dict):
            meta["llm_error"] = result.error or "LLM returned no data"
            raise RuntimeError(f"LLM decision pass failed: {meta['llm_error']}")

        decisions = result.data.get("decisions")
        if not isinstance(decisions, list):
            meta["llm_error"] = "LLM JSON has no decisions list"
            raise RuntimeError("LLM decision pass failed: JSON has no decisions list")

        decision_by_id = {
            str(decision.get("mention_id")): decision
            for decision in decisions
            if isinstance(decision, dict) and decision.get("mention_id") is not None
        }
        missing = [mention_id for mention_id in decision_concepts if mention_id not in decision_by_id]
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(f"LLM decision pass failed: missing decisions for {len(missing)} mention(s): {preview}")

        updated: list[Concept] = list(passthrough)
        for mention_id, concept in decision_concepts.items():
            decision = decision_by_id[mention_id]
            if decision.get("keep") is False:
                continue
            final_type = decision.get("final_type")
            if final_type not in ALLOWED_TYPES:
                final_type = concept.type
            raw_assertions = decision.get("assertions")
            assertions = concept.assertions
            if final_type in ASSERTION_TYPES and isinstance(raw_assertions, list):
                assertions = tuple(a for a in raw_assertions if a in ALLOWED_ASSERTIONS)
            elif final_type not in ASSERTION_TYPES:
                assertions = ()
            raw_candidates = decision.get("selected_candidates")
            candidates = concept.candidates
            if final_type in CODED_TYPES and isinstance(raw_candidates, list):
                allowed = retrieved_by_id.get(mention_id, set())
                selected = tuple(str(c) for c in raw_candidates if str(c) in allowed)
                candidates = selected
                if not candidates:
                    candidates = self.retriever.candidates_for(concept.text, final_type)
            elif final_type in CODED_TYPES and final_type != concept.type:
                candidates = self.retriever.candidates_for(concept.text, final_type)
            elif final_type not in CODED_TYPES:
                candidates = ()
            updated.append(replace(concept, type=final_type, assertions=assertions, candidates=candidates))

        meta["mode_used"] = "llm_full_doc"
        meta["llm_used"] = True
        meta["llm_decisions"] = len(decisions)
        meta["llm_decision_scope"] = "ambiguous_diagnosis_and_labs"
        meta["llm_decision_passthrough"] = len(passthrough)
        meta["llm_decision_passthrough_by_type"] = passthrough_by_type
        return sorted(updated, key=lambda c: (c.position[0], c.position[1], c.type)), meta

    def run_directory(self, input_dir: Path, output_dir: Path, limit: int | None = None) -> RunSummary:
        files = discover_input_files(input_dir)
        if limit is not None:
            files = files[: max(0, limit)]

        output_dir.mkdir(parents=True, exist_ok=True)
        timings: list[FileTiming] = []
        total_start = time.perf_counter()
        concept_total = 0
        for input_path in files:
            start = time.perf_counter()
            concepts = self.process_file(input_path)
            out_path = output_path_for(input_path, output_dir)
            write_output(out_path, concepts)
            seconds = time.perf_counter() - start
            concept_total += len(concepts)
            timings.append(
                FileTiming(
                    input_path=str(input_path),
                    output_path=str(out_path),
                    concepts=len(concepts),
                    seconds=seconds,
                )
            )
        total_seconds = time.perf_counter() - total_start
        per_file = [item.seconds for item in timings]
        return RunSummary(
            files=len(files),
            concepts=concept_total,
            total_seconds=total_seconds,
            average_seconds=(sum(per_file) / len(per_file)) if per_file else 0.0,
            p50_seconds=statistics.median(per_file) if per_file else 0.0,
            p95_seconds=_percentile(per_file, 0.95) if per_file else 0.0,
            timings=tuple(timings),
        )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = pos - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _merge_span_candidates(spans: list[SpanCandidate]) -> list[SpanCandidate]:
    priority = {
        TYPE_DRUG: 5,
        TYPE_DIAGNOSIS: 4,
        TYPE_TEST_RESULT: 3,
        TYPE_TEST_NAME: 2,
        TYPE_SYMPTOM: 1,
    }
    ordered = sorted(
        spans,
        key=lambda span: (
            -span.score,
            -(span.end - span.start),
            -priority.get(span.type, 0),
            span.start,
            span.end,
        ),
    )
    selected: list[SpanCandidate] = []
    for span in ordered:
        if span.start >= span.end:
            continue
        if any(span.start < item.end and item.start < span.end for item in selected):
            continue
        selected.append(span)
    return sorted(selected, key=lambda span: (span.start, span.end, span.type))


def _compact_candidate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "code": str(row.get("code") or ""),
            "name": str(row.get("name") or ""),
            "system": str(row.get("system") or ""),
        }
        for row in rows
        if row.get("code")
    ]


def _needs_llm_decision(concept: Concept, candidate_rows: list[dict[str, object]]) -> bool:
    if concept.type in {TYPE_TEST_NAME, TYPE_TEST_RESULT}:
        return True
    if concept.type == TYPE_DIAGNOSIS:
        return len(candidate_rows) > 1
    return False


def main(argv: list[str] | None = None) -> int:
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Run AI Race medical ontology inference")
    parser.add_argument("--input-dir", type=Path, default=paths.input_dir)
    parser.add_argument("--output-dir", type=Path, default=paths.output_dir)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    summary = MedicalKGPipeline().run_directory(args.input_dir, args.output_dir, limit=args.limit)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0
