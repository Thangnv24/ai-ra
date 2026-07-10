# Ghi Chú Nghiên Cứu

Tài liệu này giữ các hướng dữ liệu và ontology nên ưu tiên cho bài toán "Ontological Reasoning in Medical Knowledge Retrieval". Luồng LLM hiện tại chỉ dùng GPT/OpenAI-compatible API trong giai đoạn thử nghiệm.

## Nguồn Dữ Liệu Nên Ưu Tiên

- ICD-10/ICD-10-CM: dùng cho chuẩn hóa `CHẨN_ĐOÁN` sang mã bệnh.
- RxNorm/RxCUI: dùng cho chuẩn hóa `THUỐC` sang mã thuốc.
- UMLS/SNOMED CT/HPO: dùng khi có quyền truy cập để mở rộng synonym, semantic type, hierarchy và mapping.
- NCBI Disease Corpus, BC5CDR, MedMentions: dùng làm nguồn tham khảo để mở rộng alias và kiểm tra concept normalization.
- MIMIC/n2c2: chỉ dùng nếu có quyền truy cập hợp lệ; không tải tự động trong repo.
- Từ điển tiếng Việt nội bộ: ưu tiên mở rộng từ official input, tên bệnh tiếng Việt, viết tắt xét nghiệm, tên thuốc brand/generic.

## Hướng Pipeline Khuyến Nghị

1. Rule/regex/lexicon tạo mention có recall cao.
2. Section detection tách vùng bệnh sử, tiền sử gia đình, thuốc đang dùng, xét nghiệm, chẩn đoán.
3. Context/assertion detection gán `isNegated`, `isFamily`, `isHistorical`.
4. Candidate retrieval lấy top-K ICD-10/RxNorm từ local index.
5. GPT/OpenAI-compatible API chỉ chọn lại type/assertion/candidate trong dữ liệu đã truy hồi, không tự bịa mã.
6. Validator kiểm tra offset, type, assertion, candidate field trước khi ghi JSON.

## Nguyên Tắc An Toàn

- Không gửi PHI thật hoặc dữ liệu không được phép lên API.
- Không cho API sinh final JSON tự do.
- Mọi candidate được chọn phải đến từ local ontology retrieval.
- Nếu API lỗi hoặc trả JSON sai schema, fallback về baseline rule/local retrieval.
- Final output luôn là file JSON hợp lệ trong `output/`.
