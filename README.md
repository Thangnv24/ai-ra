# AI Race Medical Ontology Retrieval

Hệ thống cho bài toán **Ontological Reasoning in Medical Knowledge Retrieval**. Mục tiêu là đọc ghi chú y khoa tự do, trích xuất concept, phân loại `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, `THUỐC`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, gắn assertion, map `CHẨN_ĐOÁN` sang ICD-10 và `THUỐC` sang RxNorm, rồi ghi JSON đúng schema nộp bài.

Phiên bản hiện tại ưu tiên Track A/top 15: dùng local rule + local candidate index + LLM API mạnh khi có key. Nếu không có API key, pipeline tự fail-open về nhánh local để vẫn chạy được.

## Cài Đặt Nhanh

Yêu cầu Python `>=3.10`.

```bash
pip install -r requirements.txt
pip install -e .
copy .env.example .env
```

Trong `.env`, thông thường chỉ cần đổi dòng này:

```text
AI_RACE_API_KEY=your_api_key_here
```

Các biến còn lại trong `.env.example` đã đặt mặc định cho Track A:

```text
AI_RACE_MODE=llm_full_doc
AI_RACE_USE_LLM=true
AI_RACE_BASE_URL=https://api.openai.com/v1
AI_RACE_MODEL=gpt-4.1
AI_RACE_TEMPERATURE=0
AI_RACE_MAX_TOKENS=4096
AI_RACE_TIMEOUT=120
AI_RACE_FAIL_OPEN=true
```

`AI_RACE_*` là bộ biến môi trường duy nhất được hỗ trợ trong phiên bản hiện tại.

## Cách Chạy

Khởi động API server:

```bash
python main.py
```

Chạy toàn bộ thư mục `input/` qua API:

```bash
python tests/test.py --mode llm_full_doc
```

Mặc định kết quả được ghi vào thư mục theo ngày:

```text
output/out_put_DDMMYYYY/1.json
output/out_put_DDMMYYYY/2.json
...
```

Chạy riêng một file `.txt`:

```bash
python tests/test.py input/1.txt --mode llm_full_doc
```

Kết quả mặc định:

```text
output/out_put_DDMMYYYY/1.json
```

Chỉ định output riêng:

```bash
python tests/test.py input/1.txt --mode llm_full_doc --output-dir output/single_run
```

Runner root `test.py` dùng được cả server và direct local:

```bash
python test.py --input-dir input --out output --mode llm_full_doc
python test.py --direct --input-dir input --out output --mode llm_full_doc
python test.py --direct --file input/1.txt --out output --mode llm_full_doc
```

Tạo thư mục output riêng cho review thủ công:

```bash
python tests/test.py input --mode llm_full_doc --output-dir output/review_YYYYMMDD_HHMMSS
```

Chỉ đóng gói submission khi được yêu cầu rõ:

```bash
ai-race-submit --output-dir output/review_YYYYMMDD_HHMMSS --submission-dir submission
```

## Chạy Bằng Docker

Build image:

```bash
docker build -t ai-race-medical-kg:latest .
```

Chạy API bằng Docker Compose:

```bash
docker compose up --build ai-race-api
```

API sẽ mở tại:

```text
http://127.0.0.1:8000
```

Chạy batch direct trong container, đọc `input/` và ghi `output/`:

```bash
docker compose --profile run run --rm ai-race-runner
```

Compose tự đọc `.env` nếu file tồn tại. Nếu chưa có `AI_RACE_API_KEY`, pipeline vẫn chạy fallback local và báo `llm_used_files=0`.

## Luồng Hoạt Động

Luồng server:

```text
python main.py
  -> src/api/server.py
  -> tạo singleton MedicalKGPipeline
  -> /predict, /predict_batch, /predict_file
```

Luồng một file:

```text
input/1.txt
  -> core.io.read_text
  -> services.pipeline.MedicalKGPipeline.process_text_with_meta
  -> extraction.ner.MedicalNER.extract
  -> extraction.labs.extract_lab_spans
  -> extraction.llm_entities.LLMEntityExtractor.extract nếu mode=llm_full_doc và có API key
  -> extraction.context.ContextDetector.assertions_for
  -> knowledge.retrieval.CandidateRetriever.candidates_for
  -> integrations.openai_client.ApiLLMClient + prompts.build_decision_prompt nếu mode=hybrid/llm_full_doc và có API key
  -> knowledge.reasoning.infer_relations
  -> core.schema.validate_output
  -> output/<stem>.json
```

LLM đang được dùng ở 2 bước:

1. `llm_full_doc`: LLM đọc văn bản/chunk và đề xuất mention mới. Offset không lấy tự do từ LLM; hệ thống căn lại span theo quote trong văn bản gốc.
2. `hybrid` và `llm_full_doc`: LLM nhận danh sách mention đã có, context ngắn và candidate ICD-10/RxNorm đã retrieve sẵn. LLM chỉ được quyết định keep/drop, sửa type/assertion và chọn candidate trong danh sách cho phép.

Nếu thiếu key, API lỗi, hoặc JSON LLM không hợp lệ, `AI_RACE_FAIL_OPEN=true` giúp hệ thống giữ kết quả rule/local thay vì dừng.

## Cấu Trúc Dự Án

```text
.
|-- .env.example                 # Cấu hình mẫu, chỉ cần thay AI_RACE_API_KEY
|-- AGENTS.md                    # Quy tắc làm việc trong repo
|-- main.py                      # Entry point FastAPI
|-- test.py                      # Client thủ công: chạy qua server hoặc direct local
|-- tests/test.py                # Runner E2E qua server, output mặc định theo ngày
|-- input/                       # File .txt chính thức
|-- output/                      # JSON dự đoán
|-- submission/                  # output.zip để nộp, chỉ cập nhật khi được yêu cầu
|-- problem/                     # Đề bài, sample, scoring
|-- data/candidates/             # Candidate KB đã giản lược cho runtime
|-- scripts/                     # Script tải/build/inspect dữ liệu
|-- src/api/                     # FastAPI endpoints
|-- src/core/                    # Config, path, I/O, schema, text normalization
|-- src/extraction/              # NER, lab extractor, section/chunk, LLM entity proposal
|-- src/knowledge/               # Ontology seed, slim candidate index, retrieval, reasoning
|-- src/integrations/            # OpenAI-compatible client và prompt JSON
|-- src/services/                # Pipeline điều phối end-to-end
`-- src/submission/              # Công cụ đóng gói khi được yêu cầu
```

## Vai Trò Từng Nhóm File

`src/core/config.py` định nghĩa nhãn hợp lệ, assertion hợp lệ, đường dẫn chuẩn và biến môi trường `AI_RACE_*`.

`src/core/schema.py` chứa model concept và `validate_output()`. Mọi output phải qua bước này để kiểm tra type, position, assertion và candidate.

`src/extraction/ner.py` là rule extractor chính cho symptom, diagnosis, drug và một số pattern y khoa phổ biến.

`src/extraction/labs.py` tách tên xét nghiệm và kết quả xét nghiệm trực tiếp từ input, ví dụ `WBC:14,43`, `AST ... là 319`.

`src/extraction/sectioning.py` chia văn bản thành section/chunk ổn định để hỗ trợ LLM trên văn bản dài.

`src/extraction/llm_entities.py` gọi LLM để đề xuất entity từ chunk, sau đó align lại bằng quote trong text gốc.

`src/knowledge/candidates.py` load `data/candidates/*.jsonl`, gồm ICD-10/RxNorm đã lọc cột cần thiết và alias normalized.

`src/knowledge/ontology.py` giữ seed ontology nhỏ và lookup exact/fuzzy. Seed giúp hệ thống vẫn chạy khi candidate KB thiếu hoặc chưa build.

`src/knowledge/retrieval.py` hợp nhất kết quả từ seed ontology và slim candidate KB, trả về candidate code cho output hoặc candidate rows cho LLM rerank.

`src/integrations/prompts.py` ép LLM trả JSON có cấu trúc, không sinh output tự do.

`src/services/pipeline.py` là nơi nối toàn bộ luồng: extract, LLM proposal, assertion, retrieval, LLM decision, validate.

## Dữ Liệu Và Candidate KB

Runtime hiện dùng thư mục đã giản lược:

```text
data/candidates/icd10_candidates.jsonl
data/candidates/rxnorm_candidates.jsonl
data/candidates/candidate_aliases.jsonl
data/candidates/candidate_manifest.json
```

Các file này chỉ giữ trường cần cho truy hồi candidate: code, system, type, name, aliases, alias_norms, priority và một số cờ phụ trợ. Cách này giảm nhiễu và tránh phải đọc full RxNorm/ICD-10 rất nặng trong inference.

Script liên quan:

```bash
python scripts/build_slim_candidate_kb.py
python scripts/inspect_data_status.py
python scripts/download_icd10.py
python scripts/download_rxnorm.py
```

Các script download/build là bước chuẩn bị dữ liệu. Inference cuối nên dùng file local đã build, không download trong lúc chạy.

## Output JSON

Mỗi file output là một JSON list. Mỗi item có dạng:

```json
{
  "text": "aspirin 325mg x 1",
  "type": "THUỐC",
  "position": [2199, 2216],
  "assertions": [],
  "candidates": ["1191"]
}
```

Quy ước:

- `position` là zero-based, end-exclusive, phải khớp đúng substring trong input.
- `assertions` chỉ dùng cho `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, `THUỐC`.
- `candidates` chỉ dùng cho `CHẨN_ĐOÁN` và `THUỐC`.
- `TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM` thường trích xuất trực tiếp từ input, không cần ICD-10/RxNorm.

## Kiểm Thử

Unit tests:

```bash
python -m unittest discover -s tests
```

E2E qua server:

```bash
python main.py
python tests/test.py input/1.txt --mode llm_full_doc
python tests/test.py input --mode llm_full_doc --workers 8
```

Direct local smoke test không cần server:

```bash
python test.py --direct --input-dir input --out output --mode llm_full_doc --limit 5 --pretty
```

## Ghi Chú Về Track A Và Luật Nộp

Luồng API LLM phù hợp giai đoạn thử nghiệm/top 15 khi mục tiêu là tối ưu điểm nhanh bằng model mạnh. Trước khi nộp chính thức, cần đọc lại `problem/statement1.md`, `problem/statement2.md` và `problem/scoring.md` để xác nhận quy định dùng external API/model. Nếu thể lệ yêu cầu offline/self-host, giữ nguyên retrieval/local validation và thay LLM API bằng model nội bộ hoặc tắt `AI_RACE_USE_LLM`.
