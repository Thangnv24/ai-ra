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

### Package

```text
output/*.json
  -> scripts/validate_outputs.py
  -> scripts/package_submission.py
  -> submission/output.zip
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
- `python scripts/package_submission.py --output-dir output --submission-dir submission`

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
  -> package output.zip
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
- package `submission/output.zip`.

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
