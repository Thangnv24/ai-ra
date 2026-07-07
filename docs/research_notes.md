Mình đã lưu yêu cầu bài toán này để các lần sau bám đúng hướng: **Ontological Reasoning in Medical Knowledge Retrieval**, trọng tâm là **truy xuất/chuẩn hóa y khoa bằng model nhỏ ~9B**.

## 1. Nguồn dữ liệu chuẩn nên ưu tiên

| Nhóm                            | Nguồn                                  | Dùng cho                                               | Khả năng lấy dữ liệu                                                                                                                                                                 |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Clinical notes/EHR              | **MIMIC-IV-Note**                      | Ghi chú bác sĩ, discharge summaries, radiology reports | Tải qua PhysioNet, cần credential/DUA. Có 331,794 discharge summaries và 2,321,355 radiology reports đã de-identify. ([PhysioNet][1])                                                |
| Clinical NLP challenge          | **n2c2/i2b2 datasets**                 | NER, thuốc, adverse events, temporal relation, SDOH    | Cần DBMI portal + DUA; dữ liệu là clinical notes đã de-identify. ([N2C2][2])                                                                                                         |
| Disease normalization           | **NCBI Disease Corpus**                | Disease mention + concept normalization                | Tải công khai; 793 PubMed abstracts, 6,892 disease mentions, 790 concepts, train/dev/test. ([NCBI][3])                                                                               |
| Broad biomedical entity linking | **MedMentions**                        | UMLS concept linking, entity typing                    | GitHub/CC0; 4,392 PubMed titles/abstracts, annotated UMLS; có subset ST21pv cho information retrieval. ([GitHub][4])                                                                 |
| Drug/disease/relation           | **BC5CDR**                             | Chemical, disease, chemical-disease relation           | 1,500 PubMed articles, 4,409 chemicals, 5,818 diseases, 3,116 interactions. ([PMC][5])                                                                                               |
| Literature pre-annotation       | **PubTator Central / PubTator3**       | Disease, chemical, gene, variant, relation search      | API + bulk FTP; PubTator Central hỗ trợ XML/JSON/tab-delimited qua API/FTP. ([PMC][6]) PubTator3 có entity/relation annotations quy mô lớn và API/bulk download. ([OUP Academic][7]) |
| ICD-10                          | **CDC ICD-10-CM files**                | Ánh xạ disease → ICD-10-CM                             | CDC có PDF/XML/ZIP; FY26 April 1 2026 dùng đến 30/09/2026, FY27 dùng từ 01/10/2026. ([CDC Việt Nam][8])                                                                              |
| ICD-10 Việt Nam                 | **icd.kcb.vn**                         | Tên bệnh tiếng Việt ↔ ICD-10                           | Tra cứu song ngữ/danh mục ICD-10 tại hệ thống mã hóa lâm sàng KCB. ([ICD Việt Nam][9])                                                                                               |
| Drug normalization              | **RxNorm API**                         | Thuốc → RxCUI, ingredient, branded/clinical drug       | API chính thức của NLM; không cần license cho dữ liệu RxNorm non-proprietary, theo ToS. ([Lister Hill National Center][10])                                                          |
| Drug labels                     | **DailyMed SPL**                       | Thuốc, liều, đường dùng, cảnh báo, nhãn thuốc          | Có daily/weekly/monthly ZIP XML; DailyMed cung cấp SPL download. ([DailyMed][11])                                                                                                    |
| Ontology backbone               | **UMLS / SNOMED CT / HPO / BioPortal** | Synonym, semantic type, hierarchy, mapping, reasoning  | UMLS cần tài khoản/license miễn phí; SNOMED qua UMLS; HPO tải OBO/OWL; BioPortal có REST API và ontology mappings. ([Thư viện Quốc gia về Y tế][12])                                 |

Không nên crawl bệnh án thật hoặc dữ liệu bệnh nhân ngoài nguồn đã de-identify/được phép. Với bài thi, cách an toàn nhất là dùng **MIMIC/n2c2 làm clinical text**, **MedMentions/NCBI Disease/BC5CDR làm annotation chuẩn**, **ICD-10/RxNorm/UMLS/HPO/BioPortal làm ontology/coding system**.

## 2. Mã Python crawl/tải dữ liệu chuẩn

Cài trước:

```bash
pip install requests beautifulsoup4 pandas lxml datasets tqdm
```

### A. Tải ICD-10-CM từ CDC và đọc code descriptions

```python
import zipfile
import io
import requests
import pandas as pd

ICD10_CM_FY26_APRIL_ZIP = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/"
    "ICD10CM/2026-update/icd10cm-Code%20Descriptions-April-1-2026.zip"
)

def download_icd10cm_code_descriptions(url=ICD10_CM_FY26_APRIL_ZIP):
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    rows = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        print("Files:", z.namelist())

        # CDC ZIP thường chứa .txt mô tả code.
        txt_files = [f for f in z.namelist() if f.lower().endswith(".txt")]
        if not txt_files:
            raise RuntimeError("Không tìm thấy file .txt trong ZIP")

        with z.open(txt_files[0]) as f:
            for raw in f:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Format thường là: CODE <spaces> SHORT_DESC / LONG_DESC
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    rows.append({"code": parts[0], "description": parts[1]})

    return pd.DataFrame(rows)

icd_df = download_icd10cm_code_descriptions()
icd_df.to_csv("icd10cm_fy26_apr2026.csv", index=False)
print(icd_df.head())
```

### B. Chuẩn hóa thuốc bằng RxNorm API

```python
import requests

RXNAV = "https://rxnav.nlm.nih.gov/REST"

def rxnorm_lookup(drug_name: str):
    """
    Trả về candidate RxNorm concepts cho tên thuốc tự do.
    Ví dụ: metformin, aspirin 81 mg, amoxicillin.
    """
    r = requests.get(
        f"{RXNAV}/drugs.json",
        params={"name": drug_name},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    concepts = []
    for group in data.get("drugGroup", {}).get("conceptGroup", []):
        tty = group.get("tty")
        for c in group.get("conceptProperties", []) or []:
            concepts.append({
                "input": drug_name,
                "rxcui": c.get("rxcui"),
                "name": c.get("name"),
                "synonym": c.get("synonym"),
                "tty": tty,
            })
    return concepts

for c in rxnorm_lookup("metformin 500 mg"):
    print(c)
```

### C. Tải corpus chuẩn bằng Hugging Face Datasets

```python
from datasets import load_dataset

# Disease NER + normalization
ncbi = load_dataset("ncbi/ncbi_disease")

# Broad UMLS entity linking
# Nếu config thay đổi, chạy: load_dataset("bigbio/medmentions") để xem configs.
medmentions = load_dataset("bigbio/medmentions", "medmentions_st21pv_bigbio_kb")

# Chemical/disease/relation extraction
bc5cdr = load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_kb")

print(ncbi)
print(medmentions)
print(bc5cdr)
```

### D. Crawl PubMed abstracts bằng NCBI E-utilities

NCBI E-utilities là API công khai cho PubMed/PMC/Gene/Protein; nếu không có API key thì giới hạn 3 requests/giây, có key thì 10 requests/giây. ([NCBI][13])

```python
import time
import requests
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def pubmed_search(term, retmax=100, email="your_email@example.com", api_key=None):
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "sort": "pub date",
        "tool": "medical_ontology_retriever",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]

def pubmed_fetch_abstracts(pmids, email="your_email@example.com", api_key=None):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": "medical_ontology_retriever",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
        abstract_parts = []
        for node in article.findall(".//AbstractText"):
            abstract_parts.append("".join(node.itertext()))
        abstract = "\n".join(abstract_parts)

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
        })

    return records

pmids = pubmed_search('"hypertension"[Title/Abstract] AND "metformin"[Title/Abstract]', retmax=20)
records = pubmed_fetch_abstracts(pmids)
print(records[0])
```

### E. Lấy annotation tự động từ PubTator3 theo PMID

```python
import requests

def pubtator_export(pmids, fmt="biocjson"):
    """
    fmt: pubtator | biocxml | biocjson
    """
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/{fmt}"
    r = requests.get(url, params={"pmids": ",".join(pmids)}, timeout=60)
    r.raise_for_status()
    if fmt == "biocjson":
        return r.json()
    return r.text

annotations = pubtator_export(["31114887"], fmt="biocjson")
print(type(annotations))
```

### F. Tải HPO ontology OBO/OWL

```python
import requests

def download_file(url, out_path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

download_file("http://purl.obolibrary.org/obo/hp.obo", "hp.obo")
download_file("http://purl.obolibrary.org/obo/hp.owl", "hp.owl")
```

## 3. Pipeline đề xuất cho model nhỏ 9B

Kiến trúc nên theo hướng **hybrid retrieval + ontology reasoning + constrained LLM reranking**, không bắt 9B “nhớ” toàn bộ y khoa.

1. **Entity extraction**: dùng biomedical/clinical NER như scispaCy, medspaCy, hoặc fine-tune token classifier. scispaCy là pipeline spaCy cho biomedical/scientific/clinical text. ([allenai.github.io][14])
2. **Assertion/context**: dùng ConText/medspaCy cho phủ định, historical, hypothetical, family/experiencer. ConText được thiết kế để xác định negation, experiencer và temporal status trong clinical reports; medspaCy có contextual analysis, attribute assertion và section detection. ([PMC][15])
3. **Candidate generation**: tạo index từ UMLS/RxNorm/ICD/HPO gồm `canonical_name`, `synonyms`, `semantic_type`, `code_system`, `parents`, `relations`. Dùng BM25 + vector embedding + fuzzy match để lấy top-20 candidates.
4. **9B reranker**: model 9B chỉ chọn candidate đúng và xuất JSON có schema cố định: `mention`, `type`, `code_system`, `code`, `assertion`, `temporality`, `experiencer`, `confidence`, `evidence_span`.
5. **Ontology reasoning**: sau khi model chọn concept, dùng graph để suy luận: synonym expansion, parent/child disease, drug ingredient, brand/generic, SNOMED→ICD mapping, lab abnormality relation. BioPortal đặc biệt hữu ích vì có search, mappings, annotator và REST API. ([BioPortal][16])
6. **Relation extraction**: tách riêng quan hệ như `drug_treats_disease`, `drug_causes_ADE`, `lab_result_indicates_condition`, `symptom_associated_with_disease`, `family_history_of`. BC5CDR và PubTator3 rất hợp để bootstrapping relation extraction.
7. **Evaluation**: đo riêng NER F1, normalization accuracy@1/@5, assertion F1, relation F1, retrieval recall@k, và final exact-match theo ICD/RxCUI.

Lý do cách này hợp với model 9B: nghiên cứu gần đây cho thấy các LLM nhỏ có thể kết hợp với phần mềm normalization sẵn có để cải thiện precision/recall/F1 trong clinical concept normalization, thay vì bắt model tự sinh mã y khoa từ trí nhớ. ([arXiv][17]) Graph-RAG trong EHR cũng cho thấy lợi thế khi kết hợp graph traversal với semantic vector search cho truy xuất lâm sàng an toàn hơn. ([Frontiers][18])

## 4. Nguồn tạp chí / hội nghị nên đọc để lấy ý tưởng

Nên tìm bài với các keyword: **medical concept normalization**, **biomedical entity linking**, **UMLS entity linking**, **clinical assertion detection**, **ICD coding from clinical notes**, **RxNorm normalization**, **ontology-enhanced RAG**, **clinical knowledge graph**, **GraphRAG EHR**.

Các venue tốt:

* **JAMIA**: biomedical/health informatics, clinical care, clinical research, implementation science, public health, policy. ([amia.org][19])
* **Journal of Biomedical Informatics**: rất nhiều bài về clinical NLP, concept extraction, ontology, EHR.
* **Bioinformatics**, **Database: The Journal of Biological Databases and Curation**, **Nucleic Acids Research Database Issue**: mạnh về ontology, biomedical KG, PubTator/BioPortal/UMLS-style resources.
* **AMIA Annual Symposium**, **ACL BioNLP**, **ClinicalNLP**, **EMNLP/ACL findings**: mạnh về NER, entity linking, relation extraction, clinical NLP.
* **npj Digital Medicine**: AI/informatics trong y tế số và ứng dụng lâm sàng. ([Nature][20])
* **Scientific Data**: tìm dataset/data descriptor y khoa. ([Nature][21])

## 5. Hướng làm nhanh để thi

Bản baseline mạnh và khả thi:

**NER**: scispaCy/ClinicalBERT token classifier →
**Candidate retrieval**: BM25 + vector search trên ICD/RxNorm/UMLS aliases →
**Context**: medspaCy ConText →
**9B rerank**: chọn candidate + xuất JSON có schema →
**Graph reasoning**: NetworkX/RDFLib cho ontology edges →
**Evaluation**: exact code match + relation F1.

Điểm ăn giải thường nằm ở: **candidate recall@20 cao**, **không hallucinate code**, **bắt đúng negation/family/history**, và **giải thích được vì sao map vào ICD/RxNorm đó**.

[1]: https://physionet.org/content/mimic-iv-note/ "MIMIC-IV-Note: Deidentified free-text clinical notes v2.2"
[2]: https://n2c2.dbmi.hms.harvard.edu/data-sets "Data Sets | National NLP Clinical Challenges (n2c2)"
[3]: https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/ "
Disease Corpus
"
[4]: https://github.com/chanzuckerberg/MedMentions "GitHub - chanzuckerberg/MedMentions: A corpus of Biomedical papers annotated with mentions of UMLS entities. · GitHub"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4860626/?utm_source=chatgpt.com "BioCreative V CDR task corpus: a resource for chemical ..."
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6602571/ "
            PubTator central: automated concept annotation for biomedical full text articles - PMC
        "
[7]: https://academic.oup.com/nar/article/52/W1/W540/7640526?utm_source=chatgpt.com "PubTator 3.0: an AI-powered literature resource for unlocking ..."
[8]: https://www.cdc.gov/nchs/icd/icd-10-cm/files.html "ICD-10-CM Files | Classification of Diseases, Functioning, and Disability | CDC"
[9]: https://icd.kcb.vn/?utm_source=chatgpt.com "HTQL MÃ HOÁ LÂM SÀNG KHÁM CHỮA BỆNH"
[10]: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html "RxNorm API - APIs"
[11]: https://dailymed.nlm.nih.gov/dailymed/spl-resources.cfm "DailyMed - SPL Resources"
[12]: https://www.nlm.nih.gov/research/umls/index.html "Unified Medical Language System (UMLS)"
[13]: https://www.ncbi.nlm.nih.gov/home/develop/api/ "APIs - Develop - NCBI"
[14]: https://allenai.github.io/scispacy/ "scispacy | SpaCy models for biomedical text processing"
[15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2757457/?utm_source=chatgpt.com "Context: An Algorithm for Determining Negation, Experiencer ..."
[16]: https://bioportal.bioontology.org/ "Welcome to the NCBO BioPortal | NCBO BioPortal"
[17]: https://arxiv.org/html/2405.15122v1 "Generalizable and Scalable Multistage Biomedical Concept Normalization Leveraging Large Language Models"
[18]: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1780700/full "Frontiers | Unlocking electronic health records: a hybrid graph RAG approach to safe clinical AI for patient QA"
[19]: https://amia.org/news-publications/journals/jamia?utm_source=chatgpt.com "JAMIA:Journal of the American Medical ..."
[20]: https://www.nature.com/npjdigitalmed/aims?utm_source=chatgpt.com "Aims and scope | npj Digital Medicine"
[21]: https://www.nature.com/sdata/?utm_source=chatgpt.com "Scientific Data"
