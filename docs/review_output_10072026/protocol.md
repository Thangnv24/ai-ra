# Protocol Review Từng File

Mục tiêu: sửa output theo từng bệnh án, không sửa đại trà khi chưa đọc input.

## 1. Chọn file cần đọc

Ưu tiên theo `summary.md`, mục `Full Review Queue`.

Nhóm cần làm trước:

1. `very_sparse`, `under_extract`: khả năng thiếu entity, kéo WER xuống mạnh.
2. `drug_hints_but_no_drug`: input có thuốc nhưng output bỏ thuốc, kéo candidate score xuống.
3. `lab_names_without_results`, `lab_name_result_imbalance`: thiếu cặp tên xét nghiệm/kết quả.
4. `wide_candidates`: candidate list quá rộng, kéo Jaccard candidates xuống.
5. `suspicious_diagnosis_span`, `overlapping_same_type`: khả năng span/type bị sai.

## 2. Đọc file review

Mở `NNN.md` tương ứng trong thư mục này. Mỗi file có:

- Full input.
- Bảng output hiện tại đã sort theo `position`.
- Context quanh từng span.
- Candidate code kèm tên từ local ICD-10/RxNorm data.
- Các flag rủi ro của file.

## 3. Quy tắc sửa entity

- `text` phải là substring nguyên văn trong input.
- `position` là zero-based, end-exclusive, và phải khớp `input[start:end]`.
- Không dùng span mô tả quá dài nếu gold có khả năng chỉ lấy cụm bệnh/triệu chứng ngắn.
- Không thêm thông tin cá nhân như tuổi, tên, địa chỉ, số điện thoại.
- Với item lặp nhiều lần trong input, chỉ giữ những lần có ý nghĩa lâm sàng rõ, tránh overcount tiêu đề/lặp mô tả không cần thiết.

## 4. Quy tắc theo type

- `TRIỆU_CHỨNG`: lấy biểu hiện tự nhiên trong text, gồm triệu chứng hiện tại và triệu chứng phủ định nếu đề cập rõ.
- `CHẨN_ĐOÁN`: lấy bệnh/chẩn đoán, bệnh nền, phát hiện chẩn đoán hình ảnh nếu là bệnh thật.
- `THUỐC`: lấy tên thuốc kèm liều/dạng dùng nếu có trong cùng cụm.
- `TÊN_XÉT_NGHIỆM`: lấy tên xét nghiệm hoặc chỉ số xét nghiệm.
- `KẾT_QUẢ_XÉT_NGHIỆM`: lấy giá trị tương ứng, thường là số hoặc số kèm đơn vị.

## 5. Quy tắc assertion

- `isHistorical`: tiền sử bệnh, thuốc trước nhập viện, bệnh đã điều trị, bệnh mạn tính đã có trước.
- `isNegated`: gần các từ/cụm phủ định như không, không có, không ghi nhận, âm tính, chưa thấy.
- `isFamily`: bệnh/triệu chứng/thuốc thuộc người nhà, bố/mẹ/anh/chị/em/con.
- Không gắn assertion cho `TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM`.

## 6. Quy tắc candidate

- Chỉ có `candidates` cho `CHẨN_ĐOÁN` và `THUỐC`.
- Ưu tiên 1 candidate khi chắc chắn. Chỉ dùng 2 candidates nếu đề/ontology thực sự nhập nhằng.
- Tránh list 3-5 candidates kiểu cha/con nếu text không đủ bằng chứng, vì Jaccard candidates bị phạt.
- Với ví dụ/benchmark trong đề, ưu tiên mã BTC đã minh họa hơn exact-string thuần của ontology.

## 7. Validate sau mỗi file

Sau khi sửa file trong thư mục review mới dạng `output/review_YYYYMMDD_HHMMSS/NNN.json`, chạy:

```bash
python scripts/review_submitted_output.py --source-output output/review_YYYYMMDD_HHMMSS --formatted-dir output/review_YYYYMMDD_HHMMSS --review-dir docs/review_output_10072026
```

Lệnh này sẽ:

- Format lại JSON.
- Validate `text` và `position`.
- Cập nhật review note/summary.
- Không đóng gói thẳng vào `submission/` trong luồng review thủ công. Chỉ build `submission/output.zip` khi người dùng yêu cầu rõ.
