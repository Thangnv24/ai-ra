"""FastAPI server for the medical retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medkg.config import get_paths, get_settings
from medkg.io import read_text
from medkg.pipeline import MedicalKGPipeline
from medkg.schema import validate_output

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings(ROOT)
pipeline = MedicalKGPipeline(root=ROOT)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    id: str | None = None
    mode: str | None = None
    validate_result: bool = Field(True, alias="validate")

    class Config:
        allow_population_by_field_name = True


class BatchItem(BaseModel):
    id: str
    text: str = Field(..., min_length=1)


class BatchRequest(BaseModel):
    items: list[BatchItem]
    mode: str | None = None
    validate_result: bool = Field(True, alias="validate")

    class Config:
        allow_population_by_field_name = True


class FileRequest(BaseModel):
    path: str = Field(..., min_length=1)
    mode: str | None = None
    validate_result: bool = Field(True, alias="validate")

    class Config:
        allow_population_by_field_name = True


app = FastAPI(
    title="Medical Ontology Retrieval API",
    version="0.2.0",
    description="Deterministic/hybrid ICD-10/RxNorm concept extraction service.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    manifest_path = get_paths(ROOT).data_indexes / "kb_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            manifest = {}
    return {
        "status": "ok",
        "mode": settings.mode,
        "kb_loaded": True,
        "llm_enabled": settings.llm_enabled,
        "llm_available": _llm_available_quick() if settings.llm_enabled else False,
        "concept_count": manifest.get("concept_count", len(pipeline.index.entries)),
        "alias_count": manifest.get("alias_norm_count", 0),
    }


def _llm_available_quick() -> bool:
    try:
        with urllib.request.urlopen(f"{settings.llm_base_url.rstrip('/')}/models", timeout=2):  # noqa: S310
            return True
    except Exception:  # noqa: BLE001
        return False


def _predict_text(text: str, item_id: str | None, mode: str | None, validate_result: bool) -> dict[str, Any]:
    concepts, meta = pipeline.process_text_with_meta(text, mode=mode)
    output = [concept.to_dict() for concept in concepts]
    errors = validate_output(output, source_text=text) if validate_result else []
    if errors:
        raise HTTPException(status_code=500, detail={"errors": errors[:20]})
    return {
        "id": item_id,
        "concepts": output,
        "meta": meta,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        return _predict_text(request.text, request.id, request.mode, request.validate_result)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict_batch")
def predict_batch(request: BatchRequest) -> dict[str, Any]:
    results = [
        _predict_text(item.text, item.id, request.mode, request.validate_result)
        for item in request.items
    ]
    return {"results": results}


@app.post("/predict_file")
def predict_file(request: FileRequest) -> dict[str, Any]:
    path = Path(request.path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")
    return _predict_text(read_text(path), path.stem, request.mode, request.validate_result)
