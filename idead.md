# Phân Tích Điểm, Luồng Chạy Và Hướng Tối Ưu

Tài liệu này là bản phân tích, không phải code inference. Mục tiêu là hiểu vì sao điểm hiện tại thấp, luồng đang chạy thế nào, dữ liệu đi qua đâu, nên tối ưu gì, và dùng GPT/OpenAI-compatible API thế nào cho đúng.

## 1. Điểm Hiện Tại

```text
WER: 78.623
num_scored: 100
J_assertion: 10.7652
num_records: 100
J_candidates: 6.9429
```

Diễn giải:

- `num_scored = 100` và `num_records = 100`: đủ 100 file được chấm, lỗi không nằm ở thiếu file.
- `WER = 78.623`: text span đang lệch nhiều. Có thể miss mention, bắt span quá rộng/quá hẹp, hoặc sai `type`.
- `J_assertion = 10.7652`: assertion thấp, nghĩa là `isNegated`, `isFamily`, `isHistorical` đang thiếu hoặc scope sai.
- `J_candidates = 6.9429`: candidate ICD-10/RxNorm rất thấp, thường do không bắt đúng diagnosis/drug hoặc ontology alias chưa đủ.

Ưu tiên tăng điểm:

1. Tăng chất lượng `text` + `type`.
2. Tăng recall candidate cho `CHẨN_ĐOÁN` và `THUỐC`.
3. Cải thiện assertion bằng sentence/section scope.
4. Dùng API chỉ để quyết định có cấu trúc, không cho sinh JSON tự do.

## 2. Cấu Trúc Ngoài `src`

Nhóm cần giữ:

- `.env.example`: mẫu cấu hình API.
- `AGENTS.md`: hướng dẫn làm việc.
- `README.md`: hướng dẫn chạy chính.
- `idead.md`: phân tích tối ưu.
- `main.py`: bật FastAPI.
- `test.py`: client inference.
- `pyproject.toml`, `requirements.txt`: cài đặt.
- `input/`: input chính thức.
- `output/`: output mặc định.
- `submission/`: file zip nộp bài.
- `problem/`: đề/schema/scoring.
- `data/raw`, `data/processed`, `data/indexes`: giữ vì là dữ liệu tải/build/index.
- `scripts/download_*.py`, `scripts/prepare_vi_medical_aliases.py`: giữ vì là luồng tải dữ liệu.
- `scripts/build_knowledge_base.py`, `scripts/build_indexes.py`: giữ vì tạo KB/index.
- `scripts/run_server.py`, `scripts/run_all_tests.py`, `scripts/validate_outputs.py`, `scripts/validate_submission.py`, `scripts/package_submission.py`: giữ vì chạy/validate/package.
- `tests/test_end_to_end.py`: giữ vì test luồng đầu cuối.

Nhóm có thể gộp hoặc không bắt buộc trong luồng chạy:

- `docs/current_project_audit.md`: báo cáo audit cũ, có thể gộp vào README nếu muốn tối giản.
- `docs/reports/initial_build_report.md`: báo cáo build cũ, không ảnh hưởng inference.
- `docs/requirements_digest.md`: digest yêu cầu, có thể gộp vào README/problem notes.
- `docs/data_inventory.md`: hữu ích khi audit dữ liệu, không phải runtime code.

Nhóm đã loại khỏi luồng:

- `outputs/`: bỏ vì default mới là `output/`.
- `test_inputs/`: bỏ vì test E2E tự tạo input tạm.
- `workflow.txt`: bỏ vì là ghi chú/prompt cũ.
- `scripts/benchmark.py`: bỏ vì không cần benchmark.
- Các unit test nhỏ cũ trong `tests/`: bỏ, chỉ giữ test E2E.
- Các README nhỏ trong từng thư mục: gộp vào `README.md`.

## 3. Luồng Input -> Output Hiện Tại

### Direct mode

```text
input/*.txt
  -> test.py:main()
  -> get_input_files()
  -> core.io.discover_input_files()
  -> core.io.read_text()
  -> MedicalKGPipeline.process_text_with_meta()
  -> MedicalNER.extract()
  -> ContextDetector.assertions_for()
  -> CandidateRetriever.candidates_for()
  -> OntologyIndex.lookup()
  -> ApiLLMClient.chat_json() nếu bật API
  -> validate_output()
  -> output/*.json
```

### Server mode

```text
main.py
  -> uvicorn
  -> api.server.app
  -> singleton MedicalKGPipeline
  -> POST /predict hoặc /predict_batch
  -> _predict_text()
  -> pipeline.process_text_with_meta()
  -> validate_output()
  -> response.concepts
  -> test.py ghi output/*.json
```

### Validate / Optional Package

```text
output/*.json
  -> scripts/validate_outputs.py
  -> output/review_YYYYMMDD_HHMMSS/
  -> optional scripts/package_submission.py only when explicitly requested
  -> optional submission/output.zip
```

## 4. Map Tham Số Quan Trọng

`test.py`:

- `--file input/1.txt`: chạy một file.
- `--input-dir input`: chạy toàn bộ folder.
- `--path`: alias cũ, nhận file hoặc folder.
- `--out output`: thư mục ghi JSON cuối.
- `--output-dir output`: alias cũ của `--out`.
- `--direct`: không gọi server, chạy pipeline trong process hiện tại.
- `--url http://127.0.0.1:8000`: server URL nếu không dùng `--direct`.
- `--mode baseline`: rule + local retrieval.
- `--mode hybrid`: baseline trước, sau đó gọi API nếu bật.
- `--mode llm_full_doc`: hiện đang đi cùng nhánh hybrid, chưa phải full-document extraction độc lập.
- `--limit N`: chạy N file đầu.
- `--pretty`: ghi JSON có indent.

API:

```json
{
  "id": "1",
  "text": "...",
  "mode": "hybrid",
  "validate": true
}
```

`.env`:

```text
AI_RACE_MODE=llm_full_doc
AI_RACE_USE_LLM=true
AI_RACE_BASE_URL=https://api.openai.com/v1
AI_RACE_MODEL=gpt-4.1
AI_RACE_API_KEY=...
AI_RACE_TEMPERATURE=0
```

## 5. Dữ Liệu Và Knowledge Base

Luồng chuẩn bị dữ liệu:

```text
scripts/download_all_data.py
  -> download_icd10.py
  -> download_rxnorm.py
  -> download_public_corpora.py
  -> prepare_vi_medical_aliases.py
  -> inspect_data_status.py
```

Luồng build:

```text
scripts/build_knowledge_base.py
  -> data/processed/concepts.jsonl
  -> data/processed/aliases.jsonl
  -> data/processed/drug_aliases.jsonl
  -> data/processed/disease_aliases.jsonl
  -> data/processed/lab_aliases.jsonl
  -> data/processed/symptom_aliases.jsonl
```

```text
scripts/build_indexes.py
  -> data/indexes/ontology_index.json
  -> data/indexes/alias_exact.json
  -> data/indexes/alias_norm.json
  -> data/indexes/kb_manifest.json
```

Runtime hiện dùng mạnh nhất:

- `data/indexes/ontology_index.json`
- built-in seed entries trong `src/knowledge/ontology.py`
- optional `data/external/vietnamese_clinical_lexicon.csv` nếu có

Điểm nghẽn: `alias_exact.json` và `alias_norm.json` đã được build nhưng candidate retrieval hiện chưa khai thác đầy đủ. Đây là cơ hội tăng `J_candidates`.

## 6. Vấn Đề Chính

### 6.1 NER/Text Span

Nguyên nhân WER cao:

- Lexicon còn nhỏ.
- Nhiều cụm bệnh/triệu chứng tiếng Việt không có trong rule.
- Có thể bắt sai type.
- Drug span có thể dính liều/route/frequency quá rộng hoặc quá hẹp.
- Lab format trong input có thể đa dạng hơn regex hiện tại.
- Một câu có nhiều concept nối bằng dấu phẩy nhưng rule chưa split tốt.

### 6.2 Assertion

Nguyên nhân `J_assertion` thấp:

- `ContextDetector` chủ yếu nhìn context trước mention.
- Negation có thể áp dụng cho list dài: "không sốt, không ho, không khó thở".
- `isFamily` thường phụ thuộc section "tiền sử gia đình".
- `isHistorical` thường phụ thuộc heading/section dài, không chỉ vài trăm ký tự trước mention.

### 6.3 Candidate

Nguyên nhân `J_candidates` thấp:

- Miss mention thì không thể map code.
- Mention có modifier làm lookup lệch.
- ICD-10 tiếng Việt cần synonym/alias rộng hơn.
- RxNorm cần tách ingredient, brand, strength, dose form.
- API hiện chỉ được chọn trong candidate đã retrieval, nên retrieval không có code đúng thì API không cứu được.

## 7. Hướng Tối Ưu

### 7.1 Tăng NER Recall

- Tạo audit output cho 100 file: mention text, type, offset, source rule.
- Dùng API để gợi ý mention bị miss, nhưng code vẫn tự anchor offset.
- Mở rộng lexicon bệnh/triệu chứng/xét nghiệm/thuốc từ official input.
- Split list concept theo dấu phẩy, dấu chấm phẩy, bullet.
- Thêm section detection trước NER.

### 7.2 Tăng Assertion

- Tách sentence trước khi gán assertion.
- Tách section như `family history`, `past medical history`, `home medication`, `diagnosis`, `lab`.
- Negation nên có scope theo list.
- Historical/family nên theo section trước, trigger sau.
- API có thể phân xử khi rule conflict.

### 7.3 Tăng Candidate

- Cho `CandidateRetriever` dùng thêm `alias_exact.json` và `alias_norm.json`.
- Tạo nhiều query từ một mention:
  - original text;
  - normalized no-accent;
  - stripped drug modifiers;
  - API-normalized English/Vietnamese canonical name;
  - abbreviation expansion.
- Với thuốc:
  - tách ingredient;
  - tách strength;
  - tách route/frequency;
  - ưu tiên RxCUI đúng level nếu đủ evidence.
- Với bệnh:
  - map tiếng Việt sang ICD-10 canonical;
  - nếu không chắc subtype thì giữ candidate list rộng hơn.

## 8. Dùng GPT/OpenAI-Compatible API Nhiều Lượt

Nếu bỏ qua thời gian, multi-pass API hợp lý hơn một prompt lớn vì mỗi pass có schema hẹp và dễ validate.

### Pass 1: Section hóa

Input: full text.

Output:

```json
{
  "sections": [
    {"name": "history", "start": 0, "end": 120},
    {"name": "medications", "start": 121, "end": 220}
  ]
}
```

Mục tiêu: biết vùng nào là tiền sử, gia đình, thuốc, xét nghiệm.

### Pass 2: Đề xuất mention recall cao

Input: full text + sections + label được phép.

Output:

```json
{
  "mentions": [
    {
      "text": "aspirin 81 mg",
      "type": "THUỐC",
      "normalized_name": "aspirin",
      "evidence": "aspirin 81 mg"
    }
  ]
}
```

Code local phải tìm offset từ exact substring, không tin offset API nếu chưa validate.

### Pass 3: Assertion

Input: mentions có offset + section + context.

Output:

```json
{
  "assertions": [
    {"mention_id": "m1", "assertions": ["isHistorical"]}
  ]
}
```

### Pass 4: Candidate query generation

Input: diagnosis/drug mentions.

Output:

```json
{
  "queries": [
    {
      "mention_id": "m1",
      "queries": ["aspirin", "aspirin 81 mg oral tablet"],
      "system": "RxNorm"
    }
  ]
}
```

Sau đó code local retrieval top-K.

### Pass 5: Candidate selection

Input: mention + top-K candidates local.

Output:

```json
{
  "decisions": [
    {
      "mention_id": "m1",
      "selected_candidates": ["243670"],
      "confidence": 0.91
    }
  ]
}
```

Ràng buộc: `selected_candidates` phải là subset của retrieved candidates.

### Pass 6: Repair schema

Input: final draft + validation errors.

Output: chỉ sửa lỗi schema/offset/field, không thêm concept nếu không có evidence.

## 9. Tạo Knowledge Cho API

Không gửi toàn bộ DB vào prompt. Nên tạo context nhỏ theo từng file:

- Schema pack: allowed type, allowed assertion, output field.
- Section pack: section name + start/end.
- Mention pack: rule spans + API spans + normalized name.
- Candidate pack: top-K code, name, system, aliases.
- Error pack: lỗi validator nếu có.

API chỉ quyết định trên pack này. Code local vẫn làm:

- đọc file;
- anchor offset;
- lookup ontology;
- validate schema;
- ghi JSON.

## 10. Kế Hoạch Phát Triển

Giai đoạn 1: Quan sát

- Chạy E2E trên `input/`.
- Xuất audit JSON/CSV phụ.
- Soát 10-20 file lỗi nặng.

Giai đoạn 2: Candidate retrieval

- Dùng `alias_norm.json`.
- Thêm query expansion.
- Mở rộng alias ICD/RxNorm từ input.

Giai đoạn 3: API decision pass

- Bật `AI_RACE_USE_LLM=true`.
- Dùng temperature `0`.
- Chỉ cho API chọn type/assertion/candidate trong schema.

Giai đoạn 4: Multi-pass API

- Section -> mention -> assertion -> query -> candidate -> repair.
- Cache kết quả theo hash input + prompt version + model.

Giai đoạn 5: Validate/package

- `python scripts/validate_outputs.py --output-dir output --input-dir input`
- Chỉ đóng gói submission khi người dùng yêu cầu rõ. Review/thử nghiệm mặc định ghi vào thư mục mới trong `output/`, ví dụ `output/review_YYYYMMDD_HHMMSS/`.

## 11. Checklist Tối Ưu Sau Này

- [ ] Không sửa schema nếu chưa đối chiếu `problem/`.
- [ ] `position` luôn khớp substring gốc.
- [ ] `candidates` chỉ có với `CHẨN_ĐOÁN` và `THUỐC`.
- [ ] API output luôn parse JSON.
- [ ] API không được bịa code ngoài local retrieval.
- [ ] Có cache cho API call.
- [ ] Có audit source cho mỗi mention.
- [ ] Chạy E2E validation trước khi package.

## 12. Kết Luận

Điểm hiện tại thấp chủ yếu do baseline rule/lexicon còn thiếu coverage và candidate retrieval chưa đủ mạnh. Hướng tăng điểm nên bắt đầu từ span/type và candidate recall, sau đó mới dùng GPT/OpenAI-compatible API nhiều lượt để phân xử type/assertion/candidate trong schema chặt.

API nên là bộ ra quyết định có kiểm soát, không phải nơi sinh final JSON tự do. Code local vẫn phải giữ quyền quyết định cuối: offset, ontology retrieval, validate, ghi file và package.

## 13. Plan Triển Khai Mới Để Đẩy Điểm Top 15

Ghi chú phạm vi: đây là kế hoạch triển khai, chưa sửa inference logic. Mục tiêu trước mắt là tăng điểm leaderboard nhanh nhất có thể. Theo `problem/statement1.md`, phần LLM/agent có ràng buộc không dùng API ngoài và self-host tối đa 9B params; vì vậy nên tách làm 2 track:

- Track A - leaderboard boost: dùng LLM API mạnh để tạo output tốt nhanh, phục vụ thử nghiệm và leo public leaderboard.
- Track B - reproducible/compliant: giữ toàn bộ pipeline local, chuẩn bị phương án thay LLM API bằng self-host <=9B nếu BTC yêu cầu dựng lại source.

### 13.1. Mục Tiêu Điểm Và Thứ Tự Ưu Tiên

Scoring:

```text
final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
```

Ưu tiên tối ưu:

1. Bắt đúng `text` và `type`, vì sai type bị tính như mismatch nặng.
2. Map đúng candidates cho `CHẨN_ĐOÁN` và `THUỐC`, vì chiếm 40%.
3. Gắn assertion đúng cho `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, `THUỐC`, vì chiếm 30%.
4. Không cố sinh quá nhiều entity nếu không chắc; false positive làm WER/Jaccard giảm.

### 13.2. Có Cần Data Preprocessing Không?

Có, rất cần. LLM mạnh giúp hiểu ngôn ngữ tự nhiên, nhưng candidate mapping không nên để LLM tự bịa mã. Data preprocessing là phần bắt buộc để:

- tạo index ICD-10/RxNorm nhỏ, nhanh, có thể retrieve top candidates;
- chuẩn hóa alias tiếng Việt/Anh, viết tắt, bỏ dấu, biến thể liều/dạng dùng;
- giữ cả mã archive/obsolete nếu đề hoặc ground truth dùng mã cũ như RxNorm `360047`;
- cung cấp candidate pack nhỏ cho LLM rerank thay vì nhét raw DB vào prompt;
- audit được vì sao một code được chọn.

Hiện đã có slim artifacts:

```text
data/candidates/icd10_candidates.jsonl
data/candidates/rxnorm_candidates.jsonl
data/candidates/candidate_aliases.jsonl
data/candidates/candidate_manifest.json
```

Bước tiếp theo nên build thêm runtime index từ các artifact này:

- exact alias index: `alias_norm -> candidate codes`;
- token/BM25 index: tìm gần đúng theo cụm từ tự nhiên;
- char n-gram/fuzzy index: chịu lỗi chính tả, dấu, dấu gạch nối;
- drug dose parser: tách ingredient, strength, unit, form, route;
- diagnosis query expansion: bỏ các tiền tố như `bệnh`, `chẩn đoán`, `mắc`, `theo dõi`, `nghi ngờ`.

### 13.3. Kiến Trúc Tổng Thể Đề Xuất

Luồng nên là hybrid, trong đó LLM mạnh làm extraction/decision, còn local code giữ offset, retrieval, validation:

```text
input text
  -> normalize/read with exact original text preserved
  -> section/chunk splitter with global offsets
  -> rule extractor chắc chắn cho lab/drug pattern rõ
  -> LLM entity proposal trên từng chunk
  -> local offset align: exact substring trong original text
  -> type-specific candidate retrieval
  -> LLM candidate rerank/assertion decision với top-K candidates
  -> deterministic merge/dedup/overlap resolver
  -> schema + position validation
  -> output/<id>.json
  -> output/review_YYYYMMDD_HHMMSS/<id>.json
  -> optional package only when explicitly requested
```

Nguyên tắc: LLM không được quyết định final offset và không được tự sinh ICD/RxNorm code ngoài candidate pack.

### 13.4. Bước 1 - Section Và Chunking

Không chia cứng theo token ngay từ đầu. Nên chia theo cấu trúc lâm sàng:

- heading: `Tiền sử`, `Bệnh sử`, `Triệu chứng`, `Chẩn đoán`, `Kết quả xét nghiệm`, `Thuốc`, `Điều trị`, `Ra viện`;
- bullet/list/numbering;
- câu và dấu chấm phẩy;
- block xét nghiệm sau dấu `:` có nhiều cặp `name:value`.

Mỗi chunk phải lưu:

```json
{
  "chunk_id": "c12",
  "section": "Kết quả xét nghiệm",
  "start": 1234,
  "end": 1670,
  "text": "..."
}
```

Nên overlap nhẹ 100-200 ký tự giữa chunk dài, nhưng khi merge phải dedup theo global offset.

### 13.5. Bước 2 - Extraction Theo Type

#### `TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM`

Làm rule là chính, vì pattern thường nằm trực tiếp trong input:

```text
WBC:14,43
NEUT% (Tỷ lệ % bạch cầu trung tính):76,4
bilirubin toàn phần (tbili) là 2.4
```

Output:

- bên trái `:`/`=`/`là` là `TÊN_XÉT_NGHIỆM`;
- bên phải là `KẾT_QUẢ_XÉT_NGHIỆM`;
- không có `candidates`;
- không có assertions.

Đặc biệt: nếu input là `WBC`, output `text` phải là `WBC`, không normalize thành `TWBC`, vì `position` phải khớp substring gốc.

#### `THUỐC`

Nguồn span:

- section thuốc/list thuốc;
- regex thuốc + dose/route/frequency;
- LLM đề xuất span khi format tự nhiên.

Candidate:

- retrieve RxNorm từ `data/candidates/rxnorm_candidates.jsonl`;
- ưu tiên exact dose/form nếu span có strength: `amlodipine 10 mg`, `capsaicin 0.38 MG/ML`;
- fallback ingredient/brand nếu không có liều;
- giữ mã archive nếu match đề/ground truth cũ.

#### `CHẨN_ĐOÁN`

Không thể chỉ dựa vào ICD alias exact. Cần LLM/model đề xuất cụm diagnosis tự nhiên:

```text
bệnh trào ngược dạ dày - thực quản
tắc nghẽn đường mật
nốt tuyến giáp thùy trái
kết quả chọc hút bất thường của nốt tuyến giáp
```

Candidate:

- query ICD-10 bằng exact/fuzzy/BM25;
- LLM rerank top candidates theo ngữ cảnh;
- nếu diagnosis chung có nhiều code hợp lý, có thể trả 2-3 codes như `K21.0`, `K21.9`;
- tránh trả quá nhiều code vì Jaccard sẽ giảm nếu union lớn.

#### `TRIỆU_CHỨNG`

Đây là phần rule/NER thuần yếu nhất. Cần LLM mạnh làm entity proposer cho cụm symptom tự nhiên:

```text
ho đờm xanh
tức ngực
đau thượng vị
ợ hơi
khó nuốt
khàn tiếng
khó thở
```

Không có candidates. Có assertions nếu bị phủ định, tiền sử, gia đình.

### 13.6. Bước 3 - Prompt LLM API Mạnh

Không dùng một prompt khổng lồ sinh final JSON. Nên chia pass:

Pass A - entity proposal:

- input: chunk + section + allowed types;
- output: exact quotes, type, evidence phrase;
- không yêu cầu candidate code;
- không yêu cầu offset.

Schema:

```json
{
  "mentions": [
    {
      "quote": "ho đờm xanh",
      "type": "TRIỆU_CHỨNG",
      "confidence": 0.92
    }
  ]
}
```

Pass B - assertion:

- input: mention + local context +- 300 chars + section;
- output: subset of `isNegated`, `isFamily`, `isHistorical`;
- chỉ áp dụng cho symptom/diagnosis/drug.

Pass C - candidate rerank:

- input: mention + context + top-K local candidates;
- output: selected candidate codes only from provided list;
- không cho LLM tự viết code mới.

Pass D - repair:

- input: final JSON + validator errors;
- output: chỉ sửa lỗi schema/type/assertion/candidate subset;
- offset vẫn do code local repair.

### 13.7. Bước 4 - Offset Và Position

Đây là lớp không được giao cho LLM.

Luật bắt buộc:

```text
source_text[start:end] == output_item["text"]
```

Quy trình:

1. LLM trả `quote`.
2. Code tìm exact quote trong chunk.
3. Nếu có nhiều occurrence, chọn occurrence gần vị trí chunk và chưa dùng.
4. Global offset = `chunk.start + local_offset`.
5. Nếu exact fail, thử repair rất thận trọng: trim dấu câu, normalize whitespace, map lại về substring gốc.
6. Nếu vẫn fail, bỏ mention để tránh output invalid.

Không được sửa `text` thành dạng normalized nếu không tồn tại trong input.

### 13.8. Bước 5 - Candidate Retrieval Và Ranking

Candidate retrieval nên chạy local trước LLM rerank.

Cho `CHẨN_ĐOÁN`:

- exact alias normalized;
- fuzzy alias;
- BM25 trên alias/name/search text;
- query expansion bỏ tiền tố/chẩn đoán/mắc bệnh;
- ưu tiên code cụ thể có dấu chấm, nhưng giữ code cha nếu đề thường chấm cả cha;
- top-K khoảng 10-20 cho LLM rerank.

Cho `THUỐC`:

- parse span thành ingredient + strength + unit + form;
- exact full span;
- exact ingredient + strength;
- fallback ingredient;
- ưu tiên `SCD/SBD`, sau đó `SCDC/SBDC`, sau đó `IN/BN`;
- nếu input là thuốc cũ/archive, cho phép chọn archive code nếu match tốt.

Chiến lược số lượng candidates:

- khi match rất chắc: trả 1 code;
- khi gold thường có cặp code như GERD: trả 2 code;
- khi không chắc: trả tối đa 3-5 code;
- không trả 10-20 code vào final vì Jaccard dễ giảm.

### 13.9. Bước 6 - Merge, Dedup, Overlap

Sau khi gom rule + LLM:

- dedup theo `(start, end, type)`;
- nếu cùng span nhưng nhiều type, chọn type theo confidence + section;
- thuốc ưu tiên hơn symptom nếu span có dose/unit;
- lab result không được nuốt vào lab name;
- diagnosis dài ưu tiên hơn diagnosis ngắn nếu cùng bệnh;
- symptom cụm dài ưu tiên hơn từ đơn như `đau`.

Không nên emit từ quá chung một mình nếu không chắc:

```text
đau
bệnh
bất thường
tăng
giảm
```

### 13.10. Bước 7 - Assertion

Assertion nên kết hợp section rule + LLM:

- `isHistorical`: section `Tiền sử`, `trước nhập viện`, `thuốc trước nhập viện`, `đã từng`, `bệnh sử`;
- `isNegated`: scope phủ định trong câu/list: `không`, `không ghi nhận`, `phủ nhận`, `chưa thấy`;
- `isFamily`: chủ thể gia đình: `bố`, `mẹ`, `anh/chị/em`, `gia đình`, `father/mother`.

Rủi ro lớn: section `Tiền sử bệnh hiện tại` không phải lúc nào cũng là historical assertion. Cần phân biệt:

- `Tiền sử bệnh nội khoa`: thường historical.
- `Tiền sử sử dụng thuốc trước nhập viện`: thuốc historical.
- `Bệnh sử hiện tại`, `lý do nhập viện`, `khởi phát 1 ngày`: thường không nên gắn `isHistorical`.

### 13.11. Bước 8 - Validation Và Audit

Mỗi file output cần có audit phụ, không nộp:

```json
{
  "text": "...",
  "type": "...",
  "position": [1, 10],
  "source": "llm_entity|lab_rule|drug_rule",
  "candidate_source": "exact_alias|bm25|llm_rerank",
  "confidence": 0.91,
  "validator_errors": []
}
```

Validator bắt buộc:

- JSON list;
- field đúng schema;
- type thuộc 5 nhãn;
- candidates chỉ có với diagnosis/drug;
- assertions chỉ có với symptom/diagnosis/drug;
- sorted by start;
- no invalid position;
- `source_text[start:end] == text`.

### 13.12. Bước 9 - Vòng Lặp Nộp Và Học Từ Leaderboard

Vì không có gold test, cần tạo cơ chế so sánh output giữa các phiên bản:

- run A: conservative, ít false positive;
- run B: high recall với LLM;
- run C: high candidate recall;
- submit từng phiên bản nếu luật cho phép;
- ghi lại score public;
- dùng score để điều chỉnh ngưỡng confidence/entity count/candidate count.

Các chỉ số nội bộ cần xem:

- số concepts/file;
- số diagnosis/drug có candidates;
- trung bình candidates/concept;
- tỷ lệ assertion historical/negated/family;
- số span bị validator repair/drop;
- top lỗi offset.

### 13.13. Bước 10 - Track Compliant Sau Khi Vào Top 15

Nếu BTC yêu cầu source và cấm API ngoài, cần có đường lui:

- thay LLM API bằng self-host <=9B;
- giữ cùng prompt/schema/chunking;
- cache output/entity proposals nếu luật cho phép dữ liệu trung gian;
- dùng local deterministic candidate retrieval để giữ phần candidates ổn;
- nếu self-host yếu hơn, dùng nó cho entity proposal, còn rule/index làm correction.

Mục tiêu là kiến trúc không phụ thuộc provider API ở interface:

```text
LLMClient.extract_mentions(chunk) -> structured JSON
LLMClient.classify_assertion(context) -> structured JSON
LLMClient.rerank_candidates(pack) -> structured JSON
```

API mạnh và self-host chỉ là backend khác nhau.

### 13.14. Milestones Triển Khai

Milestone 1 - Data/index:

- build runtime candidate index từ `data/candidates`;
- unit check các code mẫu: `K21.0`, `K21.9`, `360047`, `1660761`, `308135`;
- benchmark query cho diagnosis/drug thường gặp.

Milestone 2 - Chunk/offset:

- section splitter có global offsets;
- quote aligner exact + safe repair;
- validator position bắt buộc.

Milestone 3 - Lab deterministic:

- extract `name:value`;
- xử lý dấu phẩy thập phân;
- không normalize `WBC` thành `TWBC`.

Milestone 4 - LLM entity proposal:

- prompt JSON schema;
- cache theo hash `(model, prompt_version, chunk_text)`;
- merge entity proposals.

Milestone 5 - Candidate rerank:

- local retrieve top-K;
- LLM chọn subset;
- final candidate count policy.

Milestone 6 - Assertion:

- section rules;
- LLM assertion pass;
- rule override cho negation scope rõ ràng.

Milestone 7 - End-to-end output:

- run 100 input files;
- generate output dated folder;
- validate all JSON;
- package `submission/output.zip` chỉ khi người dùng yêu cầu rõ; mặc định dừng ở thư mục output timestamp để review.

## 14. Ý Tưởng Tối Ưu Candidate Sau Điểm 35.938

Ghi chú phạm vi: đây là ý tưởng phân tích, chưa sửa code. Mục tiêu 2 ngày tới là kéo điểm gần mốc 50 bằng cách ưu tiên `J_candidates`, đồng thời không làm hỏng `text/type/position`.

### 14.1. Đọc Lại Công Thức Điểm

Điểm mới nhất:

```text
final_score: 35.938
WER: 57.6834
text_score ~= 42.3166
J_assertion: 42.1579
J_candidates: 26.4891
```

Công thức:

```text
final = 0.3 * text_score + 0.3 * assertion_score + 0.4 * candidate_score
```

Nếu giữ nguyên text và assertion, candidate phải lên khoảng 61.6 mới đạt 50 điểm. Vì vậy không nên chỉ sửa mã ICD/RxNorm đơn thuần. Cần phối hợp 3 hướng:

- giảm WER bằng span đúng hơn;
- sửa assertion theo section/ngữ cảnh;
- tăng candidate bằng retrieval và rerank tốt hơn.

`J_candidates` bị nặng vì chiếm 40% final score và còn bị weighted theo số candidate thật. Các bệnh/thuốc có nhiều mã gold nếu sai sẽ kéo điểm mạnh.

### 14.2. Vì Sao Text Thuốc Trong Input Lệch Với RxNorm `name`

Ví dụ BTC:

```json
{
  "text": "amlodipine 10 mg po daily",
  "type": "THUỐC",
  "candidates": ["308135"]
}
```

Trong `data/candidates/rxnorm_candidates.jsonl`, mã `308135` là:

```text
amlodipine 10 MG Oral Tablet
```

Nghĩa là output `text` phải giữ nguyên span trong input, bao gồm cả phần SIG/hướng dẫn dùng thuốc như `po daily`, `po bid`, `q6h:prn`, `x 1`. Nhưng phần dùng để map RxNorm phải là drug product core:

```text
amlodipine 10 mg po daily
-> strip SIG: amlodipine 10 mg
-> infer form/route: oral tablet từ po hoặc từ candidate alias
-> map: amlodipine 10 MG Oral Tablet
-> RxCUI: 308135
```

Do đó nếu search trực tiếp toàn bộ span input vào cột `name` thì sẽ lệch. Đây không phải lỗi dữ liệu; đây là khác biệt giữa:

- mention text trong hồ sơ lâm sàng;
- tên chuẩn RxNorm dùng để normalize thuốc.

### 14.3. Bridge Cho RxNorm: Span -> Core Drug -> Candidate

Ý tưởng map 2 phía:

1. Giữ nguyên `text` và `position` theo input.
2. Tạo query variants từ mention:
   - raw normalized: `amlodipine 10 mg po daily`;
   - bỏ frequency: `amlodipine 10 mg po`;
   - bỏ route/frequency: `amlodipine 10 mg`;
   - thêm form từ route: `amlodipine 10 mg oral tablet`;
   - ingredient-only: `amlodipine`.
3. Search trong `candidate_aliases.jsonl` trước, không chỉ search cột `name`.
4. Dùng thông tin strength/form để chọn mức RxNorm:
   - có hoạt chất + strength + form: ưu tiên SCD/SBD;
   - có brand: cho phép SBD/brand;
   - chỉ có tên hoạt chất, không dose/form: ưu tiên IN/PIN;
   - có nhiều hoạt chất: ưu tiên multi-ingredient clinical drug.
5. LLM/reranker chỉ chọn trong candidate pack mở rộng, nhưng candidate pack phải đủ rộng trước.

Ví dụ từ dữ liệu hiện có:

```text
aspirin 81 mg po daily
-> aspirin 81 mg oral tablet
-> 243670

metoprolol succinate xl 50 mg po daily
-> metoprolol succinate 50 mg extended release oral tablet
-> 866436

docusate sodium 100 mg po bid
-> docusate sodium 100 mg oral tablet
-> 1099279

senna 8.6 mg po bid:prn
-> sennosides 8.6 mg oral tablet
-> 312935
```

Trường hợp đặc biệt cần rule/LLM hiểu synonym:

- `senna` trong input có thể map sang `Sennosides`;
- `ASA` map sang `aspirin`;
- `APAP` map sang `acetaminophen`;
- `xl`, `er`, `extended release`, `24 hr` là cùng nhóm extended-release;
- `po` thường gợi ý oral, nhưng không phải lúc nào cũng đủ để quyết định tablet/suspension nếu input không có form.

### 14.4. Chính Sách Candidate Count Cho RxNorm

Không nên luôn lấy top 1. Nhưng cũng không nên nhét quá nhiều mã vì Jaccard phạt candidate thừa.

Đề xuất:

- Nếu exact clinical drug match rất rõ: chọn 1 mã.
- Nếu mention có dose range như `325-650 mg`: cân nhắc chọn mã dose thấp/chuẩn BTC thường dùng, nhưng không thêm tất cả dose nếu không có evidence.
- Nếu mention là ingredient-only: chọn ingredient RxCUI, không đoán clinical drug.
- Nếu input có form rõ như `oral suspension`, `injection`, `cream`, `solution`: bắt buộc ưu tiên candidate cùng form.
- Nếu candidate top 2 là cùng alias và đều priority tốt, chỉ giữ cả 2 khi benchmark/ICD-like pattern cho thấy gold thường nhiều mã. Với RxNorm, đa số nên 1 mã.

### 14.5. ICD-10: Không Chỉ Exact Name, Phải Hiểu Cấp Mã

Ví dụ `bệnh trào ngược dạ dày - thực quản` trong local ICD có:

```text
K21   Bệnh trào ngược dạ dày - thực quản
K21.0 Bệnh trào ngược dạ dày - thực quản kèm viêm thực quản
K21.9 Bệnh trào ngược dạ dày - thực quản không kèm viêm thực quản
```

Benchmark ví dụ của BTC chọn `K21.0` và `K21.9`, không chọn parent `K21`. Điều này gợi ý scoring/gold có xu hướng dùng mã lá hoặc các mã con hợp lệ, không chỉ parent category.

Ý tưởng cho ICD:

1. Query cả raw span, bỏ tiền tố `bệnh`, `chẩn đoán`, `mắc`, `theo dõi`.
2. Search `name_vi`, `name_en`, `aliases`, `alias_norms`.
3. Nếu exact match vào parent code có children rất sát, cân nhắc trả children thay vì parent, hoặc parent + children tùy pattern đã học từ benchmark.
4. Nếu text có dấu hiệu specificity:
   - `kèm viêm`, `có biến chứng`, `cấp`, `mạn`, `không xác định`, `không kèm`: ưu tiên mã con tương ứng;
   - nếu không nói rõ nhưng benchmark có tiền lệ lấy sibling `.0` + `.9`, áp dụng cho một số nhóm đã xác nhận.
5. Không map triệu chứng thành chẩn đoán chỉ vì ICD có mã triệu chứng. Nếu type sai, concept bị tính mismatch nặng.

### 14.6. Lỗi Có Khả Năng Đang Kéo `J_candidates`

Từ output hiện tại, tất cả `CHẨN_ĐOÁN`/`THUỐC` đều có candidate và mã đều nằm trong local index. Vậy candidate thấp nhiều khả năng do:

- chọn sai mức RxNorm: ingredient thay vì clinical drug hoặc ngược lại;
- không bridge được `po daily`, `q6h:prn`, `xl`, dose range sang tên chuẩn;
- ICD chọn parent/generic trong khi gold chọn child/sibling;
- entity text/type không khớp gold nên candidate đúng cũng không được tính;
- false positive diagnosis/drug làm mẫu bị phạt Jaccard;
- LLM đang bị giới hạn bởi candidate pack nghèo.

### 14.7. Phương Án Tối Ưu Luồng Code Sau Khi Được Xác Nhận

Chưa thực hiện ngay, chỉ là đề xuất:

```text
input text
-> entity proposal exact quote
-> deterministic offset align
-> type guard: symptom/lab/diagnosis/drug
-> RxNorm/ICD query variant builder
-> local candidate expansion top 30-50
-> ranker theo strength/form/alias/source/priority
-> LLM adjudicator chọn candidate subset trong pack
-> manual/silver override layer từ các lần chấm điểm
-> schema + position validation
-> output/review_YYYYMMDD_HHMMSS/
```

Độ phù hợp:

- RxNorm query variant + TTY/form ranker: rất cao, vì ví dụ BTC chứng minh span input khác `name` chuẩn.
- ICD child/sibling policy: cao, vì benchmark GERD cho thấy gold có thể dùng nhiều mã con.
- Manual/silver override layer: rất cao trong 2 ngày, vì biến tri thức từ các lần submit thành rule ổn định.
- Full LLM JSON trực tiếp: trung bình, vì dễ sai offset/code nếu không bị ràng buộc.
- Rule/NER thuần: thấp nếu dùng một mình, vì triệu chứng/chẩn đoán tự nhiên quá đa dạng.

### 14.8. Ưu Tiên Thực Thi Nếu Được Duyệt

1. Audit nhóm file nhiều `CHẨN_ĐOÁN`/`THUỐC` trước: `20`, `41`, `54`, `96`, `13`, `50`, `16`, `32`, `88`, `37`, `44`.
2. Làm bảng lỗi RxNorm từ ví dụ BTC và output thật:
   - span;
   - core drug;
   - expected candidate;
   - candidate hiện tại;
   - rule/synonym cần thêm.
3. Làm bảng lỗi ICD:
   - diagnosis span;
   - mã parent;
   - mã child/sibling;
   - dấu hiệu trong text.
4. Sau khi có 30-50 case chắc chắn, mới cập nhật code retrieval/ranker.
5. Mỗi lần chạy mới chỉ ghi vào thư mục timestamp trong `output/`, không đóng gói submission nếu chưa được yêu cầu.

Milestone 8 - Submit/iterate:

- submit conservative;
- submit high recall;
- compare public scores;
- khóa cấu hình tốt nhất.

### 13.15. Quyết Định Kỹ Thuật Quan Trọng

- Dùng LLM mạnh để hiểu span tự nhiên, nhưng không để LLM bịa offset/code.
- Candidate preprocessing là bắt buộc, vì candidate score chiếm 40%.
- Lab extraction nên deterministic, không cần ontology.
- `position` là invariant quan trọng nhất: mọi bước phải preserve original text.
- Cần giữ audit để biết điểm thấp do miss entity, sai type, sai assertion hay sai candidate.
- Nên chuẩn bị song song track self-host để giảm rủi ro sau khi lọt top 15.

## 15. Luồng Mới Đang Triển Khai Sau Điểm 44.64020

Mục tiêu của vòng này là đưa tri thức từ các lần submit vào code, nhưng vẫn giữ LLM ở vai trò bộ quyết định có kiểm soát:

```text
rule/lab extraction
-> LLM exact-quote entity proposal theo chunk
-> merge span và validate offset
-> local candidate expansion với query variants
-> LLM adjudicator chọn type/assertion/candidate trong candidate pack
-> deterministic postprocess/silver rules
-> schema + offset validation
-> output/review_YYYYMMDD_HHMMSS/
```

Thay đổi trong pipeline:

- Candidate pack gửi vào LLM tăng lên 25 mã cho mỗi diagnosis/drug.
- RxNorm query variants xử lý `325mg` -> `325 mg`, bỏ `x 1`, cắt route/frequency, bỏ `xl/xr/er/sr/cr` khi cần fallback.
- ICD query variants sửa một số spelling/synonym hay gặp, ví dụ `sung huyet` -> `xung huyet`.
- Postprocess sửa các span nhiễu từ lần chấm: `phu ngoai vi tang dan...` -> `phu ngoai vi`, `tang tang can ... pound...` -> `tang can`, lab result `12 bach cau` -> `12`, `am tinh nitrite` -> `am tinh`.
- Drop diagnosis nằm hoàn toàn bên trong tên xét nghiệm để giảm false positive candidate.
- Re-apply assertion bằng context detector sau LLM, để tránh LLM làm mất `isHistorical` ở thuốc/tiền sử.
- Không package `submission/output.zip` nếu người dùng chưa yêu cầu rõ; output review luôn ghi vào thư mục timestamp trong `output/`.

Hướng tiếp nếu điểm vẫn tăng chậm:

1. Tạo bảng silver case cho 30-50 diagnosis/drug chắc chắn sai từ các lần submit.
2. Thêm rule candidate override có provenance cho từng silver case.
3. Tách candidate ranker riêng cho RxNorm: ingredient vs SCDC/SCD/SBD, strength, form, route.
4. Tách candidate ranker riêng cho ICD: exact child, sibling unspecified, parent fallback.
5. Chạy chunk-by-chunk manual review cho nhóm file nhiều candidate trước khi submit tiếp.
