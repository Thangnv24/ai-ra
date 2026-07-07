"""Configuration and competition constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


TYPE_SYMPTOM = "TRI\u1ec6U_CH\u1ee8NG"
TYPE_TEST_NAME = "T\u00caN_X\u00c9T_NGHI\u1ec6M"
TYPE_TEST_RESULT = "K\u1ebeT_QU\u1ea2_X\u00c9T_NGHI\u1ec6M"
TYPE_DIAGNOSIS = "CH\u1ea8N_\u0110O\u00c1N"
TYPE_DRUG = "THU\u1ed0C"

ALLOWED_TYPES = (
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
)

ASSERTION_NEGATED = "isNegated"
ASSERTION_FAMILY = "isFamily"
ASSERTION_HISTORICAL = "isHistorical"

ALLOWED_ASSERTIONS = (
    ASSERTION_NEGATED,
    ASSERTION_FAMILY,
    ASSERTION_HISTORICAL,
)

ASSERTION_TYPES = {
    TYPE_SYMPTOM,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
}

CODED_TYPES = {
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
}

SYSTEM_ICD10 = "ICD10"
SYSTEM_RXNORM = "RxNorm"


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_external: Path
    data_processed: Path
    data_indexes: Path
    index_file: Path
    input_dir: Path
    output_dir: Path
    submission_dir: Path


@dataclass(frozen=True)
class Settings:
    mode: str
    llm_enabled: bool
    llm_backend: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout: int
    llm_fail_open: bool


def project_root() -> Path:
    env_root = os.environ.get("MEDKG_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def get_paths(root: Path | None = None) -> Paths:
    base = (root or project_root()).resolve()
    default_input = base / "input"
    if not default_input.exists():
        default_input = base / "test_inputs"
    return Paths(
        root=base,
        data_raw=base / "data" / "raw",
        data_external=base / "data" / "external",
        data_processed=base / "data" / "processed",
        data_indexes=base / "data" / "indexes",
        index_file=base / "data" / "indexes" / "ontology_index.json",
        input_dir=default_input,
        output_dir=base / "outputs",
        submission_dir=base / "submission",
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (project_root() / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_settings(root: Path | None = None) -> Settings:
    base = (root or project_root()).resolve()
    load_dotenv(base / ".env")
    return Settings(
        mode=os.environ.get("MEDKG_MODE", "hybrid"),
        llm_enabled=_env_bool("MEDKG_LLM_ENABLED", False),
        llm_backend=os.environ.get("MEDKG_LLM_BACKEND", "openai_compatible"),
        llm_base_url=os.environ.get("MEDKG_LLM_BASE_URL", "http://localhost:8000/v1"),
        llm_model=os.environ.get("MEDKG_LLM_MODEL", "local-9b-medical"),
        llm_api_key=os.environ.get("MEDKG_LLM_API_KEY", "dummy"),
        llm_temperature=_env_float("MEDKG_LLM_TEMPERATURE", 0.0),
        llm_max_tokens=_env_int("MEDKG_LLM_MAX_TOKENS", 2048),
        llm_timeout=_env_int("MEDKG_LLM_TIMEOUT", 60),
        llm_fail_open=_env_bool("MEDKG_LLM_FAIL_OPEN", True),
    )
