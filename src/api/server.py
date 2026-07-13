"""FastAPI server for the medical retrieval pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from core.config import get_paths, get_settings
from core.io import read_text
from core.schema import validate_output
from services.pipeline import MedicalKGPipeline

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings(ROOT)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("ai_race").setLevel(LOG_LEVEL)
logger = logging.getLogger("ai_race.api")
pipeline = MedicalKGPipeline(root=ROOT)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class BatchItem(BaseModel):
    text: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class BatchRequest(BaseModel):
    items: list[BatchItem]

    class Config:
        extra = "forbid"


class FileRequest(BaseModel):
    path: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


app = FastAPI(
    title="Medical Ontology Retrieval API",
    version="0.2.0",
    description="LLM-required ICD-10/RxNorm concept extraction service.",
)


@app.middleware("http")
async def log_http_request(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        seconds = time.perf_counter() - start
        logger.exception(
            "request_failed request_id=%s method=%s path=%s seconds=%.6f",
            request_id,
            request.method,
            request.url.path,
            seconds,
        )
        raise
    seconds = time.perf_counter() - start
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s seconds=%.6f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        seconds,
    )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    paths = get_paths(ROOT)
    manifest_path = paths.data_indexes / "kb_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            manifest = {}
    verified_index = paths.index_file.exists() and bool(manifest)
    return {
        "status": "ok",
        "mode": settings.mode,
        "kb_loaded": verified_index,
        "knowledge_source": "verified_index" if verified_index else "built_in_seed",
        "llm_required": True,
        "llm_available": _llm_available_quick(),
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "concept_count": manifest.get("concept_count", len(pipeline.index.entries)),
        "alias_count": manifest.get("alias_norm_count", 0),
    }


def _llm_available_quick() -> bool:
    start = time.perf_counter()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{settings.llm_base_url.rstrip('/')}/models", timeout=2):  # noqa: S310
            logger.info(
                "llm_health_check_ok base_url=%s seconds=%.6f",
                settings.llm_base_url,
                time.perf_counter() - start,
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "llm_health_check_failed base_url=%s seconds=%.6f error=%s",
            settings.llm_base_url,
            time.perf_counter() - start,
            exc,
        )
        return False


def _predict_text(text: str) -> dict[str, Any]:
    start = time.perf_counter()
    logger.info("predict_start chars=%s mode=llm_full_doc model=%s", len(text), settings.llm_model)
    concepts, meta = pipeline.process_text_with_meta(text)
    output = [concept.to_dict() for concept in concepts]
    errors = validate_output(output, source_text=text)
    if errors:
        logger.error("predict_validation_failed errors=%s seconds=%.6f", errors[:20], time.perf_counter() - start)
        raise HTTPException(status_code=500, detail={"errors": errors[:20]})
    seconds = time.perf_counter() - start
    logger.info(
        "predict_complete chars=%s concepts=%s llm_used=%s seconds=%.6f",
        len(text),
        len(output),
        meta.get("llm_used"),
        seconds,
    )
    return {
        "concepts": output,
        "meta": meta,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        return _predict_text(request.text)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict_batch")
def predict_batch(request: BatchRequest) -> dict[str, Any]:
    results = [_predict_text(item.text) for item in request.items]
    return {"results": results}


@app.post("/predict_file")
def predict_file(request: FileRequest) -> dict[str, Any]:
    path = Path(request.path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")
    return _predict_text(read_text(path))
