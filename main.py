from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from api.server import app  # noqa: E402,F401


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=False)
