FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AI_RACE_MODE=llm_full_doc \
    AI_RACE_USE_LLM=true \
    AI_RACE_BASE_URL=https://api.openai.com/v1 \
    AI_RACE_MODEL=gpt-4.1 \
    AI_RACE_TEMPERATURE=0 \
    AI_RACE_MAX_TOKENS=4096 \
    AI_RACE_TIMEOUT=120 \
    AI_RACE_FAIL_OPEN=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e .

COPY . .

RUN mkdir -p input output submission data/candidates

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
