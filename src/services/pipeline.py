"""End-to-end LLM-required inference pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.config import (
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
from extraction.assertion_model import AssertionClassifier
from extraction.llm_entities import LLMEntityExtractor
from extraction.annotation_memory import AnnotationMemory
from extraction.learned_models import SpanAcceptanceModel, TokenSpanModel
from extraction.ner import MedicalNER, SpanCandidate
from extraction.span_grammar import SpanGrammar
from extraction.span_verifier import SpanTypeVerifier
from integrations.openai_client import ApiLLMClient
from knowledge.candidates import load_slim_candidate_index
from knowledge.ontology import OntologyIndex, load_ontology_index
from knowledge.reasoning import infer_relations
from knowledge.retrieval import CandidateQueryAliases, CandidateRetriever
from services.postprocess import refine_concepts


logger = logging.getLogger("ai_race.pipeline")


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


@dataclass(frozen=True, slots=True)
class MergeSummary:
    inputs: int
    selected: int
    invalid: int
    exact_duplicates: int
    overlap_conflicts: int

    def to_dict(self) -> dict[str, int]:
        return {
            "inputs": self.inputs,
            "selected": self.selected,
            "invalid": self.invalid,
            "exact_duplicates": self.exact_duplicates,
            "overlap_conflicts": self.overlap_conflicts,
        }


class MedicalKGPipeline:
    def __init__(self, index: OntologyIndex | None = None, root: Path | None = None):
        self.root = root
        paths = get_paths(root)
        self.settings = get_settings(root)
        self._ensure_indexes(paths)
        self.index = index or load_ontology_index(paths.index_file, paths.data_raw, paths.data_external)
        self.slim_candidate_index = load_slim_candidate_index(paths.root / "data" / "candidates")
        self.ner = MedicalNER()
        self.annotation_memory = AnnotationMemory.load(paths.data_external / "annotation_memory.jsonl")
        self.token_span_model = TokenSpanModel.load(paths.data_external / "token_span_model.json.gz")
        self.span_grammar = SpanGrammar.load(paths.data_external / "span_grammar.json")
        # The learned acceptance artifact is trained on input_part2 templates.
        # Keep it out of runtime decisions until it passes cross-domain validation.
        self.span_acceptance_model = SpanAcceptanceModel.empty()
        self.span_verifier = SpanTypeVerifier(self.annotation_memory)
        # Rule assertions generalize across templates better than the
        # input_part2-trained lexical classifier.
        self.assertion_classifier = AssertionClassifier.empty()
        self.context = ContextDetector(self.assertion_classifier)
        self.candidate_query_aliases = CandidateQueryAliases.load(paths.data_external / "candidate_query_aliases.json")
        self.retriever = CandidateRetriever(
            self.index,
            self.slim_candidate_index,
            self.annotation_memory,
            self.candidate_query_aliases,
        )
        self.llm = ApiLLMClient(self.settings)
        self.llm_entity_extractor = LLMEntityExtractor(self.llm, self.annotation_memory)

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
        pipeline_start = time.perf_counter()
        if mode is not None and mode != "llm_full_doc":
            raise ValueError("Only llm_full_doc mode is supported; local fallback modes were removed")
        requested_mode = "llm_full_doc"
        if not self.settings.llm_enabled:
            raise RuntimeError("LLM is required but disabled")

        stages: dict[str, dict[str, object]] = {}

        stage_start = time.perf_counter()
        rule_spans = self.ner.propose(text)
        stages["rule_proposal"] = {
            "spans": len(rule_spans),
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info("pipeline_stage stage=rule_proposal spans=%s seconds=%.6f", len(rule_spans), time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        memory_spans = self.annotation_memory.propose(text)
        stages["memory_proposal"] = {
            "spans": len(memory_spans),
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info(
            "pipeline_stage stage=memory_proposal spans=%s seconds=%.6f",
            len(memory_spans),
            time.perf_counter() - stage_start,
        )

        stage_start = time.perf_counter()
        sequence_spans = self.token_span_model.propose(text)
        stages["sequence_proposal"] = {
            "raw_spans": len(sequence_spans),
            "spans": 0,
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info(
            "pipeline_stage stage=sequence_proposal raw_spans=%s seconds=%.6f",
            len(sequence_spans),
            time.perf_counter() - stage_start,
        )

        stage_start = time.perf_counter()
        llm_spans, summary = self.llm_entity_extractor.extract(text)
        llm_entity_meta = summary.to_dict()
        stages["llm_entity_proposal"] = {
            "spans": len(llm_spans),
            "summary": llm_entity_meta,
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info(
            "pipeline_stage stage=llm_entity_proposal chunks=%s mentions=%s aligned=%s rejected=%s deduplicated=%s seconds=%.6f",
            summary.chunks,
            summary.mentions,
            summary.aligned,
            llm_entity_meta["rejected"],
            summary.deduplicated,
            time.perf_counter() - stage_start,
        )

        grammar_spans, grammar_summary = self.span_grammar.expand(
            text,
            [*rule_spans, *memory_spans, *llm_spans],
        )
        stages["span_grammar"] = grammar_summary.to_dict()
        corroborated_sequence_spans = _corroborated_sequence_spans(
            sequence_spans,
            grammar_spans,
        )
        stages["sequence_proposal"]["spans"] = len(corroborated_sequence_spans)

        stage_start = time.perf_counter()
        proposal_lattice = [
            *grammar_spans,
            *corroborated_sequence_spans,
        ]
        stages["boundary_variants"] = {
            "inputs": len(proposal_lattice),
            "outputs": len(proposal_lattice),
            "generated": 0,
            "enabled": False,
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info(
            "pipeline_stage stage=boundary_variants enabled=false inputs=%s outputs=%s generated=0 seconds=%.6f",
            len(proposal_lattice),
            len(proposal_lattice),
            time.perf_counter() - stage_start,
        )

        stage_start = time.perf_counter()
        spans, verification_summary = self.span_verifier.select(
            text,
            proposal_lattice,
        )
        stages["span_verification"] = {
            **verification_summary.to_dict(),
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info(
            "pipeline_stage stage=span_verification inputs=%s selected=%s invalid=%s exact_duplicates=%s "
            "type_conflicts=%s below_threshold=%s overlap_conflicts=%s seconds=%.6f",
            verification_summary.inputs,
            verification_summary.selected,
            verification_summary.invalid,
            verification_summary.exact_duplicates,
            verification_summary.type_conflicts,
            verification_summary.below_threshold,
            verification_summary.overlap_conflicts,
            time.perf_counter() - stage_start,
        )

        stage_start = time.perf_counter()
        concepts: list[Concept] = []
        candidate_sources: Counter[str] = Counter()
        for span in spans:
            assertions = self.context.assertions_for(text, span.start, span.end, span.type)
            candidate_decision = self.retriever.candidate_decision_for(
                span.text,
                span.type,
                source_text=text,
                start=span.start,
                end=span.end,
            )
            candidate_sources[candidate_decision.source] += 1
            concepts.append(
                Concept(
                    text=span.text,
                    type=span.type,
                    position=(span.start, span.end),
                    assertions=assertions,
                    candidates=candidate_decision.codes,
                )
            )
        concepts = sorted(concepts, key=lambda c: (c.position[0], c.position[1], c.type))
        stages["concept_build"] = {
            "concepts": len(concepts),
            "candidate_sources": dict(sorted(candidate_sources.items())),
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info("pipeline_stage stage=concept_build concepts=%s seconds=%.6f", len(concepts), time.perf_counter() - stage_start)
        meta: dict[str, object] = {
            "mode": requested_mode,
            "mode_used": "llm_full_doc",
            "llm_required": True,
            "llm_used": True,
            "final_llm_used": False,
            "llm_error": None,
            "llm_entity": llm_entity_meta,
            "stages": stages,
        }

        stage_start = time.perf_counter()
        before_postprocess = len(concepts)
        concepts = refine_concepts(text, concepts, retriever=self.retriever, context_detector=self.context)
        meta["postprocess"] = {
            "input_concepts": before_postprocess,
            "output_concepts": len(concepts),
        }
        stages["postprocess"] = {
            "input_concepts": before_postprocess,
            "output_concepts": len(concepts),
            "dropped": before_postprocess - len(concepts),
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        logger.info(
            "pipeline_stage stage=postprocess input_concepts=%s output_concepts=%s dropped=%s seconds=%.6f",
            before_postprocess,
            len(concepts),
            before_postprocess - len(concepts),
            time.perf_counter() - stage_start,
        )
        # Keep relation inference available without changing the public schema.
        infer_relations(concepts)
        stage_start = time.perf_counter()
        errors = validate_output([concept.to_dict() for concept in concepts], source_text=text)
        if errors:
            raise ValueError("pipeline generated invalid output: " + "; ".join(errors[:5]))
        stages["validation"] = {
            "errors": 0,
            "seconds": round(time.perf_counter() - stage_start, 6),
        }
        meta["total_seconds"] = round(time.perf_counter() - pipeline_start, 6)
        logger.info(
            "pipeline_complete chars=%s concepts=%s seconds=%.6f",
            len(text),
            len(concepts),
            time.perf_counter() - pipeline_start,
        )
        return concepts, meta

    def process_file(self, input_path: Path) -> list[Concept]:
        return self.process_text(read_text(input_path))

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


def _corroborated_sequence_spans(
    sequence_spans: list[SpanCandidate],
    trusted_spans: list[SpanCandidate],
) -> list[SpanCandidate]:
    """Use token-model spans only as exact corroboration, never discovery."""

    trusted_keys = {(span.start, span.end, span.type) for span in trusted_spans}
    return [
        span
        for span in sequence_spans
        if (span.start, span.end, span.type) in trusted_keys
    ]


def _merge_span_candidates(spans: list[SpanCandidate]) -> list[SpanCandidate]:
    selected, _ = _merge_span_candidates_with_summary(spans)
    return selected


def _merge_span_candidates_with_summary(spans: list[SpanCandidate]) -> tuple[list[SpanCandidate], MergeSummary]:
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
            -int(span.source == "rule"),
            -span.score,
            (span.end - span.start) if span.source == "llm" else -(span.end - span.start),
            -priority.get(span.type, 0),
            span.start,
            span.end,
        ),
    )
    selected: list[SpanCandidate] = []
    invalid = 0
    exact_duplicates = 0
    overlap_conflicts = 0
    for span in ordered:
        if span.start >= span.end:
            invalid += 1
            continue
        conflicts = [item for item in selected if span.start < item.end and item.start < span.end]
        if conflicts:
            if any(
                span.start == item.start and span.end == item.end and span.type == item.type
                for item in conflicts
            ):
                exact_duplicates += 1
            else:
                overlap_conflicts += 1
            continue
        selected.append(span)
    ordered_selected = sorted(selected, key=lambda span: (span.start, span.end, span.type))
    return ordered_selected, MergeSummary(
        inputs=len(spans),
        selected=len(ordered_selected),
        invalid=invalid,
        exact_duplicates=exact_duplicates,
        overlap_conflicts=overlap_conflicts,
    )


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
