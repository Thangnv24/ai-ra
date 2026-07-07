# AI Race 2026 - Cuộc đua AI cho kỹ sư Việt Nam

[Quay lại đề bài](/contests/medical-2026)

Vòng 1Đang mở

# Vòng 1 - Sơ loại

02/07/2026 - 30/07/2026

[Lịch sử nộp bài](/contests/medical-2026/phases/019e649f-4e5d-70ed-b221-7a10f537281e/submissions)[Nộp bài](/contests/medical-2026/phases/019e649f-4e5d-70ed-b221-7a10f537281e/submit)

## Đề bài & Quy định

## Thể thức

-   Vòng 1, các thí sinh dự thi nộp kết quả dự đoán dưới dạng file JSON theo đúng format do Ban Tổ chức (BTC) quy định. File nộp bao gồm một file _**output.zip**_ có cấu trúc sau khi giải nén như sau:
    
    ```
    output/
        ├── 1.json     # Nhãn của bản ghi 1
        ├── 2.json     # Nhãn của bản ghi 2
        ├── …
        └── 100.json
    ```
    
    Chi tiết dạng json trong output sẽ được nêu ở ví dụ dưới.
    
-   Lưu ý:
    
    -   Trước khi vòng 1 kết thúc, BTC yêu cầu top ~15 đội gửi trước source code riêng để thực hiện dựng lại và đánh giá trên dữ liệu private test. Việc này nhằm tránh tình trạng gian lận nộp file hard code output với input được cung cấp.
    -   Source code bao gồm:
        -   tất cả các file code của nhóm (data processing, training, inference, …)
        -   data nhóm sử dụng
        -   model weights
        -   1 file readme hướng dẫn cài đặt
    -   Nếu BTC không thể cài đặt được code của nhóm thi, nhóm thi sẽ được liên lạc riêng để hỗ trợ trong 1 khoảng thời gian nhất định. Nếu nhóm không thể cung cấp hỗ trợ kịp thời sẽ bị loại.
-   VD input-output vòng 1:
    
    -   **Input:**
        
        > _'Danh sách thuốc trước nhập viện chính xác và đầy đủ._ _1\. amlodipine 10 mg po daily_ _2\. aspirin 81 mg po daily_ _3\. metoprolol succinate xl 50 mg po daily_ _4\. guaifenesin ml po q6h:prn điều trị ho_ _5\. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức_ _6\. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau_ _7\. pravastatin 40 mg po daily_ _8\. docusate sodium 100 mg po bid điều trị táo bón_ _9\. senna 8.6 mg po bid:prn điều trị táo bón_ _10\. clonazepam 0.5 mg po qam:prn điều trị lo âu_ _11\. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ'_
        
    -   **Output:**
        
        ```json
        [
          {
            "text": "amlodipine 10 mg po daily",
            "type": "THUỐC",
            "candidates": ["308135"],
            "assertions": ["isHistorical"],
            "position": [58, 83]
          },
          {
            "text": "aspirin 81 mg po daily",
            "type": "THUỐC",
            "candidates": ["243670"],
            "assertions": ["isHistorical"],
            "position": [89, 111]
          },
          {
            "text": "metoprolol succinate xl 50 mg po daily",
            "type": "THUỐC",
            "candidates": ["866436"],
            "assertions": ["isHistorical"],
            "position": [117, 155]
          },
          {
            "text": "guaifenesin ml po q6h:prn",
            "type": "THUỐC",
            "candidates": ["392085"],
            "assertions": ["isHistorical"],
            "position": [161, 186]
          },
          {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [196, 198]
          },
          {
            "text": "nystatin oral suspension 5 ml po qid:prn",
            "type": "THUỐC",
            "candidates": ["7597"],
            "assertions": ["isHistorical"],
            "position": [204, 244]
          },
          {
            "text": "đau nhức",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [254, 262]
          },
          {
            "text": "acetaminophen 325-650 mg po q6h:prn",
            "type": "THUỐC",
            "candidates": ["313782"],
            "assertions": ["isHistorical"],
            "position": [268, 303]
          },
          {
            "text": "sốt đau",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [313, 320]
          },
          {
            "text": "pravastatin 40 mg po daily",
            "type": "THUỐC",
            "candidates": ["904475"],
            "assertions": ["isHistorical"],
            "position": [326, 352]
          },
          {
            "text": "docusate sodium 100 mg po bid",
            "type": "THUỐC",
            "candidates": ["1099279"],
            "assertions": ["isHistorical"],
            "position": [358, 387]
          },
          {
            "text": "táo bón",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [397, 404]
          },
          {
            "text": "senna 8.6 mg po bid:prn",
            "type": "THUỐC",
            "candidates": ["312935"],
            "assertions": ["isHistorical"],
            "position": [410, 433]
          },
          {
            "text": "táo bón",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [443, 450]
          },
          {
            "text": "clonazepam 0.5 mg po qam:prn",
            "type": "THUỐC",
            "candidates": ["197527"],
            "assertions": ["isHistorical"],
            "position": [457, 485]
          },
          {
            "text": "lo âu",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [495, 500]
          },
          {
            "text": "clonazepam 1.5 mg po qhs",
            "type": "THUỐC",
            "candidates": ["197528"],
            "assertions": ["isHistorical"],
            "position": [507, 531]
          },
          {
            "text": "lo âu",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [541, 546]
          },
          {
            "text": "mất ngủ",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [547, 554]
          }
        ]
        ```
        

## Metric đánh giá

Kết quả của thí sinh sẽ được tính trên tập test theo các metric sau:

-   Xét theo xác định tên khái niệm: sử dụng Word Error Rate (WER) trên trường text.
-   Xét theo xác định các assertions giữa khái niệm: sử dụng metric là độ tương đồng Jaccard (Jaccard similarity) với các bệnh, thuốc và triệu chứng tương ứng, lấy trung bình tất cả các giá trị này thành 1 điểm J(assertion)
-   Xét theo xác định candidates trong khái niệm: sử dụng metric giống với xác định assertion
-   Kết quả cuối cùng được tính điểm theo công thức:

final\_score\=0.3⋅text\_score+0.3⋅assertions\_score+0.4⋅candidates\_score\\text{final\\\_score} = 0.3 \\cdot \\text{text\\\_score} + 0.3 \\cdot \\text{assertions\\\_score} + 0.4 \\cdot \\text{candidates\\\_score}final\_score\=0.3⋅text\_score+0.3⋅assertions\_score+0.4⋅candidates\_score

Trong đó, với mỗi iii là 1 sample trong tập test, mỗi kkk là 1 candidate trong sample iii, WER(i)\\text{WER}(i)WER(i) là WER của trường text trong sample iii, ground\_truth(k)\\text{ground\\\_truth}(k)ground\_truth(k), prediction(k)\\text{prediction}(k)prediction(k) lần lượt là tập ground truth, prediction của candidate kkk trong sample iii, JX(i)J\_X(i)JX​(i) là độ tương đồng Jaccard của sample iii xét trên trường XXX tương ứng của output:

text\_score\=∑i∈test(1−WER(i))len(test)\\text{text\\\_score} = \\frac{\\sum\_{i \\in \\text{test}} \\big(1 - \\text{WER}(i)\\big)}{\\text{len}(\\text{test})}text\_score\=len(test)∑i∈test​(1−WER(i))​

assertions\_score\=∑i∈testJassertions(i)len(test)\\text{assertions\\\_score} = \\frac{\\sum\_{i \\in \\text{test}} J\_{\\text{assertions}}(i)}{\\text{len}(\\text{test})}assertions\_score\=len(test)∑i∈test​Jassertions​(i)​

candidates\_score\=∑i∈testJcandidates(i)⋅(∑k∈i(len(ground\_truth(k))+1))∑i∈test∑k∈i(len(ground\_truth(k))+1)\\text{candidates\\\_score} = \\frac{\\sum\_{i \\in \\text{test}} J\_{\\text{candidates}}(i) \\cdot \\Big(\\sum\_{k \\in i} \\big(\\text{len}(\\text{ground\\\_truth}(k)) + 1\\big)\\Big)}{\\sum\_{i \\in \\text{test}} \\sum\_{k \\in i} \\big(\\text{len}(\\text{ground\\\_truth}(k)) + 1\\big)}candidates\_score\=∑i∈test​∑k∈i​(len(ground\_truth(k))+1)∑i∈test​Jcandidates​(i)⋅(∑k∈i​(len(ground\_truth(k))+1))​

JX(i)\=1 neˆˊu len(ground\_truthX(i))\=0 vaˋ len(predictionX(i))\=0J\_X(i) = 1 \\text{ nếu } \\text{len}(\\text{ground\\\_truth}\_X(i)) = 0 \\text{ và } \\text{len}(\\text{prediction}\_X(i)) = 0JX​(i)\=1 neˆˊu len(ground\_truthX​(i))\=0 vaˋ len(predictionX​(i))\=0

JX(i)\=0 neˆˊu len(ground\_truthX(i))\=0 vaˋ len(predictionX(i))≠0J\_X(i) = 0 \\text{ nếu } \\text{len}(\\text{ground\\\_truth}\_X(i)) = 0 \\text{ và } \\text{len}(\\text{prediction}\_X(i)) \\neq 0JX​(i)\=0 neˆˊu len(ground\_truthX​(i))\=0 vaˋ len(predictionX​(i))\=0

JX(i)\=∣ground\_truthX(i)∩predictionX(i)∣∣ground\_truthX(i)∪predictionX(i)∣ trong caˊc trường hợp coˋn lạiJ\_X(i) = \\frac{\\big|\\text{ground\\\_truth}\_X(i) \\cap \\text{prediction}\_X(i)\\big|}{\\big|\\text{ground\\\_truth}\_X(i) \\cup \\text{prediction}\_X(i)\\big|} \\text{ trong các trường hợp còn lại}JX​(i)\=​ground\_truthX​(i)∪predictionX​(i)​​ground\_truthX​(i)∩predictionX​(i)​​ trong caˊc trường hợp coˋn lại

-   Lưu ý: Trong trường hợp đoán đúng phần text của khái niệm nhưng sai loại (VD: đoán `CHẨN_ĐOÁN` nhưng ground truth là `TRIỆU_CHỨNG`), khái niệm sẽ bị tính 2 lần (do tạo ra 1 khái niệm mới so với ground truth) và mỗi lần đều được tính 0 điểm với cả 3 loại metric.

## Tài nguyên

Cấu hình máy được sử dụng:

-   Thí sinh tự chuẩn bị tài nguyên tính toán. Tuy nhiên, với những giải pháp LLM/agent chỉ cho phép thí sinh self-host model mà không được sử dụng API ngoài, model self-host có độ lớn tối đa là 9B params.