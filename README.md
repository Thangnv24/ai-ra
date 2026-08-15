# AI Race Medical Ontology Retrieval

Pipeline cho bài **Ontological Reasoning in Medical Knowledge Retrieval**: đọc clinical text tiếng Việt/Anh, trích xuất concept, gán type/assertion, map chẩn đoán sang ICD-10 và thuốc sang RxNorm, rồi ghi JSON đúng schema trong `problem/`.

## Cài đặt

```bash
pip install -r requirements.txt
pip install -e .
```

Mặc định hiện tại chỉ chạy luồng LLM qua API tương thích OpenAI. Sao chép `.env.example` thành `.env`, sau đó điền endpoint, model và API key của bạn. File `.env` chứa thông tin bí mật và không được commit:

```text
API_KEY=replace_with_your_api_key
BASE_URL=https://your-openai-compatible-endpoint.example/v1
MODEL=your_model_name
TEMPERATURE=0
MAX_TOKENS=4096
TIMEOUT=120
LOG_LEVEL=INFO
```

## Chạy

Server:

```bash
python main.py
```

Chạy toàn bộ `input/` qua server, output mặc định vào `output/out_put_DDMMYYYY/`:

```bash
python tests/test.py
```

Chạy một file:

```bash
python tests/test.py input/1.txt
```

Chạy song song với số worker tùy chỉnh:

```bash
python tests/test.py input --workers 8
```

Payload `/predict` chỉ nhận trường `text`:

```json
{"text": "clinical text..."}
```

Nếu LLM/API lỗi, request fail và server ghi log lỗi; hệ thống không quay về local fallback.

## Manual gold workflow

Sau khi chạy LLM và có một thư mục output mới, tạo bản gold draft để sửa tay:

```bash
python scripts/gold_workflow.py init --prediction-dir output/out_put_DDMMYYYY --gold-dir data/gold_manual/review_YYYYMMDD_HHMMSS --pretty
```

Sau khi sửa các file JSON trong `data/gold_manual/review_...`, chấm lại output:

```bash
python scripts/gold_workflow.py score --prediction-dir output/out_put_DDMMYYYY --gold-dir data/gold_manual/review_YYYYMMDD_HHMMSS
```

Script sẽ sinh report tổng và file `.errors.jsonl` để gom lỗi span, assertion, candidate, missing và extra.

## Luồng chính

```text
input/*.txt
  -> tests/test.py
  -> FastAPI /predict
  -> services.pipeline.MedicalKGPipeline
  -> extraction.MedicalNER + LLMEntityExtractor
  -> extraction.ContextDetector
  -> knowledge.CandidateRetriever với data/candidates
  -> integrations.ApiLLMClient decision pass
  -> services.postprocess.refine_concepts
  -> core.schema.validate_output
  -> output/*.json
```

LLM chỉ được trả JSON có ràng buộc; offset được align lại từ text gốc, candidate ICD-10/RxNorm chỉ chọn trong danh sách local đã retrieve.

## File quan trọng

- `problem/`: đề bài, schema, scoring, sample chính thức.
- `input/`: file `.txt` chính thức.
- `output/`: JSON dự đoán.
- `data/candidates/`: KB runtime đã slim cho ICD-10/RxNorm.
- `tests/test.py`: runner gọi local server song song và validate JSON trước khi ghi output.
- `src/services/pipeline.py`: orchestration end-to-end.
- `src/core/schema.py`: validation schema/offset.
- `src/integrations/openai_client.py`: OpenAI-compatible client cho LLM nội bộ.

`data/candidates/candidate_manifest.json` là metadata của KB: nguồn build, số record, danh sách artifact. Runtime hiện không cần đọc file này, nhưng nên giữ để audit provenance của ICD-10/RxNorm candidate.

Zip nộp bài tự tạo thủ công từ thư mục output mong muốn; repo không còn module sinh `submission/output.zip`.
