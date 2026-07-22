from __future__ import annotations

import json
import re
from pathlib import Path

from core.config import ASSERTION_TYPES, CODED_TYPES
from core.schema import validate_output


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output" / "review_20260722_164803_input_manual"

SYMPTOM = "TRIỆU_CHỨNG"
TEST = "TÊN_XÉT_NGHIỆM"
RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
DIAGNOSIS = "CHẨN_ĐOÁN"
DRUG = "THUỐC"


def concept(text: str, start: int, end: int, type_: str, assertions=(), candidates=()):
    item = {
        "text": text[start:end],
        "type": type_,
        "assertions": list(assertions),
        "position": [start, end],
    }
    if type_ in CODED_TYPES:
        item["candidates"] = list(candidates)
    return item


def rebuild(file_id: int, specs: list[dict]) -> None:
    text = (INPUT_DIR / f"{file_id}.txt").read_text(encoding="utf-8")
    output = []
    occupied: list[tuple[int, int]] = []
    for spec in specs:
        matches = list(re.finditer(re.escape(spec["text"]), text, flags=re.IGNORECASE))
        selected = spec.get("occurrences")
        if selected is not None:
            matches = [matches[index] for index in selected]
        for match in matches:
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in occupied):
                continue
            output.append(
                concept(
                    text,
                    start,
                    end,
                    spec["type"],
                    spec.get("assertions", ()),
                    spec.get("candidates", ()),
                )
            )
            occupied.append((start, end))
    output.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
    validate_output(output, text)
    (OUTPUT_DIR / f"{file_id}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mutate(
    file_id: int,
    *,
    remove_positions: set[tuple[int, int]] | None = None,
    updates: dict[tuple[int, int], dict] | None = None,
    additions: list[dict] | None = None,
) -> None:
    text = (INPUT_DIR / f"{file_id}.txt").read_text(encoding="utf-8")
    path = OUTPUT_DIR / f"{file_id}.json"
    output = json.loads(path.read_text(encoding="utf-8"))
    remove_positions = remove_positions or set()
    updates = updates or {}
    output = [item for item in output if tuple(item["position"]) not in remove_positions]
    for item in output:
        update = updates.get(tuple(item["position"]))
        if update:
            item.update(update)
            if item["type"] not in {DIAGNOSIS, DRUG}:
                item["candidates"] = []
    occupied = [(item["position"][0], item["position"][1]) for item in output]
    for spec in additions or []:
        matches = list(re.finditer(re.escape(spec["text"]), text, flags=re.IGNORECASE))
        selected = spec.get("occurrences")
        if selected is not None:
            matches = [matches[index] for index in selected]
        for match in matches:
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in occupied):
                continue
            output.append(
                concept(
                    text,
                    start,
                    end,
                    spec["type"],
                    spec.get("assertions", ()),
                    spec.get("candidates", ()),
                )
            )
            occupied.append((start, end))
    output.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
    validate_output(output, text)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_by_text(file_id: int, updates: dict[str, dict]) -> None:
    path = OUTPUT_DIR / f"{file_id}.json"
    text = (INPUT_DIR / f"{file_id}.txt").read_text(encoding="utf-8")
    output = json.loads(path.read_text(encoding="utf-8"))
    normalized_updates = {key.casefold(): value for key, value in updates.items()}
    for item in output:
        update = normalized_updates.get(item["text"].strip().casefold())
        if update:
            item.update(update)
            if item["type"] not in {DIAGNOSIS, DRUG}:
                item["candidates"] = []
    validate_output(output, text)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_candidate_ids() -> None:
    replacements = {
        "J30.9": "J30.4",
        "R56.9": "R56.8",
        "E85.81": "E85.8",
        "K58.9": "K58",
        "K20.9": "K20",
        "Z96.649": "Z96.6",
        "T86.12": "T86.1",
        "C80.1": "C80",
        "B99.9": "B99",
        "D68.59": "D68.5",
        "K59.09": "K59.0",
    }
    unsupported = {"I31.4"}
    for file_id in range(1, 101):
        path = OUTPUT_DIR / f"{file_id}.json"
        text = (INPUT_DIR / f"{file_id}.txt").read_text(encoding="utf-8")
        output = json.loads(path.read_text(encoding="utf-8"))
        for item in output:
            if item["type"] in CODED_TYPES:
                item["candidates"] = [
                    replacements.get(candidate_id, candidate_id)
                    for candidate_id in item.get("candidates", [])
                    if candidate_id not in unsupported
                ]
            else:
                item.pop("candidates", None)
            if item["type"] not in ASSERTION_TYPES:
                item["assertions"] = []
        errors = validate_output(output, text)
        if errors:
            raise ValueError(f"{file_id}.json failed validation: {errors}")
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def curate_file_1() -> None:
    rebuild(
        1,
        [
            {"text": "xét nghiệm sàng lọc trước sinh và sau sinh", "type": TEST},
            {"text": "xét nghiệm thiếu men G6PD", "type": TEST},
            {"text": "thiếu máu do tan huyết", "type": DIAGNOSIS, "candidates": ["D59.9"]},
            {"text": "chậm phát triển trí tuệ và vận động", "type": DIAGNOSIS},
            {"text": "hồng cầu bị phá hủy hàng loạt", "type": SYMPTOM},
            {"text": "hồng cầu rất dễ bị phá hủy", "type": SYMPTOM},
            {"text": "nhiễm khuẩn tiết niệu", "type": DIAGNOSIS, "candidates": ["N39.0"]},
            {"text": "thiếu máu tan huyết", "type": DIAGNOSIS, "candidates": ["D59.9"]},
            {"text": "tổn thương thần kinh", "type": SYMPTOM},
            {"text": "chậm phát triển trí tuệ", "type": DIAGNOSIS, "candidates": ["F79"]},
            {"text": "thiếu hụt men G6PD", "type": DIAGNOSIS, "candidates": ["D55.0"]},
            {"text": "xét nghiệm sàng lọc", "type": TEST},
            {"text": "Thiếu men G6PD", "type": DIAGNOSIS, "candidates": ["D55.0"]},
            {"text": "rối loạn vận động", "type": DIAGNOSIS, "candidates": ["G25.9"]},
            {"text": "suy thận cấp", "type": DIAGNOSIS, "candidates": ["N17.9"]},
            {"text": "xét nghiệm máu", "type": TEST},
            {"text": "vàng da nặng", "type": SYMPTOM},
            {"text": "Sốt cao", "type": SYMPTOM},
            {"text": "Tim đập nhanh", "type": SYMPTOM},
            {"text": "khó thở", "type": SYMPTOM},
            {"text": "vàng da", "type": SYMPTOM},
            {"text": "vàng mắt", "type": SYMPTOM},
            {"text": "thiếu máu", "type": DIAGNOSIS, "candidates": ["D64.9"]},
            {"text": "bại não", "type": DIAGNOSIS, "candidates": ["G80.9"]},
            {"text": "Nhiễm khuẩn", "type": DIAGNOSIS, "candidates": ["A49.9"]},
            {"text": "nhiễm virus", "type": DIAGNOSIS, "candidates": ["B34.9"]},
            {"text": "sốt rét", "type": DIAGNOSIS, "candidates": ["B54"]},
            {"text": "Vitamin K", "type": DRUG, "candidates": ["11258"]},
        ],
    )


def curate_file_2() -> None:
    rebuild(
        2,
        [
            {"text": "viêm lan tỏa hệ mạch máu nhỏ và vừa", "type": DIAGNOSIS},
            {"text": "Bong da đầu ngón tay, ngón chân", "type": SYMPTOM},
            {"text": "phình giãn động mạch vành", "type": DIAGNOSIS, "candidates": ["I25.4"]},
            {"text": "Sưng, đỏ mu bàn tay – chân", "type": SYMPTOM},
            {"text": "Đỏ gan bàn tay – chân", "type": SYMPTOM},
            {"text": "Lưỡi đỏ như dâu tây", "type": SYMPTOM},
            {"text": "Bệnh đa xơ cứng", "type": DIAGNOSIS, "assertions": ["isNegated"], "candidates": ["G35"]},
            {"text": "Ảo giác do rượu", "type": DIAGNOSIS, "candidates": ["F10.5"]},
            {"text": "thuốc ức chế miễn dịch", "type": DRUG},
            {"text": "hẹp tắc mạch vành", "type": DIAGNOSIS},
            {"text": "sốt cao kéo dài", "type": SYMPTOM},
            {"text": "Viêm kết mạc 2 bên", "type": SYMPTOM},
            {"text": "Ban đỏ toàn thân", "type": SYMPTOM},
            {"text": "môi đỏ – nứt", "type": SYMPTOM},
            {"text": "lưỡi đỏ dâu tây", "type": SYMPTOM},
            {"text": "biến chứng động mạch vành", "type": DIAGNOSIS},
            {"text": "nhồi máu cơ tim", "type": DIAGNOSIS, "candidates": ["I21.9"]},
            {"text": "viêm mạch máu", "type": DIAGNOSIS, "candidates": ["I77.6"]},
            {"text": "Bệnh Kawasaki", "type": DIAGNOSIS, "candidates": ["M30.3"]},
            {"text": "Kawasaki", "type": DIAGNOSIS, "candidates": ["M30.3"], "occurrences": [3, 7]},
            {"text": "phát ban toàn thân", "type": SYMPTOM},
            {"text": "sốt cấp kéo dài", "type": SYMPTOM},
            {"text": "sưng hạch cổ", "type": SYMPTOM},
            {"text": "môi đỏ, nứt", "type": SYMPTOM},
            {"text": "Lưỡi đỏ như dâu tây", "type": SYMPTOM},
            {"text": "Mắt đỏ", "type": SYMPTOM},
            {"text": "Họng đỏ", "type": SYMPTOM},
            {"text": "mảng đỏ", "type": SYMPTOM},
            {"text": "đỏ mắt", "type": SYMPTOM},
            {"text": "ban đỏ", "type": SYMPTOM},
            {"text": "viêm tim", "type": DIAGNOSIS},
            {"text": "đột tử", "type": DIAGNOSIS},
            {"text": "suy tim", "type": DIAGNOSIS, "candidates": ["I50.9"]},
            {"text": "Nhiễm khuẩn", "type": DIAGNOSIS, "candidates": ["A49.9"]},
            {"text": "nhiễm virus", "type": DIAGNOSIS, "candidates": ["B34.9"]},
            {"text": "39–40°C", "type": RESULT},
            {"text": "Sốt", "type": SYMPTOM, "occurrences": [1, 3]},
            {"text": "dát", "type": SYMPTOM},
            {"text": "sẩn", "type": SYMPTOM},
            {"text": "loạn thần", "type": DIAGNOSIS, "assertions": ["isNegated"]},
            {"text": "Công thức máu", "type": TEST},
            {"text": "CRP", "type": TEST},
            {"text": "máu lắng", "type": TEST},
            {"text": "Men gan", "type": TEST},
            {"text": "albumin", "type": TEST},
            {"text": "Xét nghiệm nước tiểu", "type": TEST},
            {"text": "Cấy máu", "type": TEST},
            {"text": "Siêu âm tim", "type": TEST},
            {"text": "ECG", "type": TEST},
            {"text": "điện tâm đồ", "type": TEST},
            {"text": "ASA", "type": DRUG, "candidates": ["1191"]},
            {"text": "huyết khối", "type": DIAGNOSIS},
        ],
    )


def curate_file_3() -> None:
    mutate(
        3,
        remove_positions={
            (660, 663),
            (1889, 1897),
            (1902, 1910),
            (2009, 2020),
            (3703, 3724),
        },
        updates={
            (520, 529): {"assertions": []},
            (2061, 2071): {"assertions": ["isHistorical"]},
            (2126, 2133): {
                "type": DIAGNOSIS,
                "assertions": ["isNegated", "isHistorical"],
                "candidates": ["R56.9"],
            },
        },
        additions=[
            {"text": "đau thắt ngực ổn định", "type": DIAGNOSIS, "candidates": ["I20.9"]},
        ],
    )


def curate_file_4() -> None:
    path = OUTPUT_DIR / "4.json"
    text = (INPUT_DIR / "4.txt").read_text(encoding="utf-8")
    output = json.loads(path.read_text(encoding="utf-8"))
    diagnosis_codes = {
        "viêm dạ dày ruột do virus": ["A08.4"],
        "hội chứng ruột kích thích": ["K58.9"],
        "loét tá tràng": ["K26.9"],
        "loét tá tràng và hồi tràng": ["K26.9"],
        "viêm thực quản độ c": ["K20.9"],
        "ung thư tuyến giáp": ["C73"],
    }
    for item in output:
        key = item["text"].strip().casefold()
        if item["type"] == DIAGNOSIS and key in diagnosis_codes:
            item["candidates"] = diagnosis_codes[key]
        if item["type"] == DRUG and key == "omeprazole":
            item["candidates"] = ["7646"]
        if tuple(item["position"]) == (527, 538):
            item["type"] = RESULT
            item["candidates"] = []
    validate_output(output, text)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def curate_file_5() -> None:
    mutate(
        5,
        remove_positions={
            (333, 343),
            (468, 478),
            (519, 529),
            (588, 598),
            (2222, 2241),
            (2266, 2309),
        },
        updates={
            (238, 281): {"candidates": ["C22.1"]},
            (1021, 1029): {"assertions": []},
            (1031, 1038): {"assertions": []},
            (1169, 1177): {"assertions": ["isNegated"]},
            (1179, 1182): {"assertions": ["isNegated"]},
            (1184, 1187): {"assertions": ["isNegated"]},
            (1189, 1196): {"assertions": ["isNegated"]},
            (1203, 1213): {"assertions": ["isNegated"]},
        },
        additions=[
            {"text": "xét nghiệm tinh dịch đồ", "type": TEST},
            {"text": "chụp tử cung vòi trứng", "type": TEST},
        ],
    )


def curate_file_6() -> None:
    mutate(
        6,
        remove_positions={
            (1166, 1174),
            (1175, 1183),
            (1185, 1206),
            (1354, 1374),
            (1438, 1448),
            (1612, 1637),
            (1802, 1810),
            (1822, 1840),
            (2063, 2075),
            (2246, 2254),
        },
    )


def curate_file_7() -> None:
    mutate(
        7,
        remove_positions={
            (138, 141),
            (659, 662),
            (691, 699),
            (1087, 1090),
            (1161, 1164),
            (1249, 1257),
            (1336, 1339),
            (1491, 1494),
            (1875, 1885),
            (2334, 2352),
            (2677, 2702),
            (3022, 3032),
        },
        updates={
            (578, 605): {"assertions": ["isHistorical"]},
            (760, 768): {"assertions": ["isHistorical"]},
            (3167, 3178): {"assertions": []},
        },
        additions=[
            {"text": "ợ nóng", "type": SYMPTOM},
        ],
    )


def curate_file_8() -> None:
    mutate(
        8,
        remove_positions={
            (48, 50),
            (507, 515),
            (1685, 1701),
            (1997, 2015),
            (2175, 2181),
            (2185, 2186),
            (2187, 2219),
            (2281, 2284),
            (2294, 2297),
            (2298, 2306),
            (2524, 2560),
            (2622, 2631),
            (2801, 2813),
            (2921, 2950),
            (2955, 2975),
            (2977, 2980),
            (2982, 2984),
            (3029, 3031),
            (3033, 3040),
        },
        updates={
            (2260, 2268): {"candidates": ["M30.3"]},
            (2403, 2411): {"candidates": ["M30.3"]},
            (2464, 2477): {"candidates": ["M30.3"]},
            (2661, 2678): {"candidates": ["I25.9"]},
            (2848, 2861): {"candidates": ["M30.3"]},
        },
        additions=[
            {"text": "đánh giá thần kinh", "type": TEST},
            {"text": "hạch >1,5 cm, chắc", "type": SYMPTOM},
            {"text": "hóa mủ", "type": SYMPTOM, "assertions": ["isNegated"]},
            {"text": "sốt siêu vi", "type": DIAGNOSIS},
            {"text": "sốt phát ban", "type": DIAGNOSIS},
            {"text": "sốt cao 3–4 ngày", "type": SYMPTOM},
            {"text": "Thiếu máu cơ tim", "type": DIAGNOSIS, "candidates": ["I25.9"]},
            {"text": "Tăng men gan", "type": RESULT},
            {"text": "Sốt ≥5 ngày", "type": SYMPTOM},
            {"text": "Viêm kết mạc 2 bên", "type": SYMPTOM},
            {"text": "ghèn", "type": SYMPTOM, "assertions": ["isNegated"]},
            {"text": "Môi – miệng thay đổi (nứt, đỏ, lưỡi dâu tây)", "type": SYMPTOM},
            {"text": "Tổn thương đầu chi (phù, đỏ, bong da)", "type": SYMPTOM},
        ],
    )


def curate_file_9() -> None:
    mutate(
        9,
        remove_positions={
            (138, 141),
            (659, 662),
            (691, 699),
            (1087, 1090),
            (1161, 1164),
            (1249, 1257),
            (1336, 1339),
            (1491, 1494),
            (1875, 1885),
            (2334, 2352),
        },
        updates={
            (573, 605): {"assertions": ["isHistorical"]},
            (760, 768): {"assertions": ["isHistorical"]},
            (2762, 2765): {"assertions": ["isHistorical"]},
        },
        additions=[
            {"text": "ợ nóng", "type": SYMPTOM},
        ],
    )


def curate_file_10() -> None:
    mutate(
        10,
        remove_positions={
            (266, 278),
            (606, 614),
            (1075, 1083),
        },
    )


def curate_files_11_to_25() -> None:
    mutate(
        11,
        remove_positions={
            (1166, 1174), (1175, 1183), (1354, 1374), (1442, 1448),
            (1802, 1810), (1822, 1840), (2133, 2146), (2306, 2319),
        },
    )
    update_by_text(
        12,
        {
            "HSV": {"candidates": ["B00.9"]},
            "Varicella Zoster Virus": {"candidates": ["B02.9"]},
        },
    )
    for file_id in (13, 16, 20):
        update_by_text(
            file_id,
            {
                "bệnh dại": {"candidates": ["A82.9"]},
                "Bệnh dại": {"candidates": ["A82.9"]},
            },
        )
    mutate(
        13,
        remove_positions={(855, 865), (1070, 1078), (1147, 1154), (1282, 1289), (1505, 1515), (2343, 2353)},
    )
    mutate(
        14,
        remove_positions={
            (346, 363), (476, 483), (1339, 1341), (1358, 1380),
            (1386, 1394), (1420, 1428), (1433, 1448), (1508, 1510),
            (1545, 1569), (1718, 1741), (1791, 1813), (1815, 1823),
            (1828, 1839), (1896, 1906),
        },
        updates={
            (1318, 1332): {"type": RESULT},
            (1576, 1605): {"type": RESULT},
            (1693, 1717): {"type": TEST},
        },
        additions=[
            {"text": "dị ứng", "type": DIAGNOSIS, "assertions": ["isNegated"]},
        ],
    )
    mutate(15, remove_positions={(1699, 1713), (1715, 1728)})
    update_by_text(15, {"Pimperan": {"candidates": ["6915"]}})
    mutate(
        16,
        remove_positions={
            (1081, 1089), (1159, 1166), (1294, 1301), (2266, 2298),
            (2299, 2333), (2334, 2345), (2507, 2539),
        },
    )
    mutate(
        17,
        remove_positions={(1324, 1331), (1362, 1368), (2084, 2114)},
    )
    update_by_text(
        17,
        {
            "Viêm nha chu": {"candidates": ["K05.3"]},
            "viêm nha chu": {"candidates": ["K05.3"]},
            "sâu răng": {"candidates": ["K02.9"]},
            "loãng xương": {"candidates": ["M81.9"]},
            "Tiểu đường": {"candidates": ["E11.9"]},
            "tiểu đường": {"candidates": ["E11.9"]},
            "trầm cảm": {"candidates": ["F32.9"]},
        },
    )
    mutate(
        18,
        remove_positions={(254, 263), (404, 407), (459, 466), (1020, 1023), (1473, 1490), (2151, 2170), (2171, 2187)},
        updates={
            (190, 218): {"type": TEST},
            (791, 805): {"candidates": ["I25.1"]},
        },
        additions=[
            {"text": "thiếu máu cơ tim", "type": DIAGNOSIS, "candidates": ["I25.9"]},
        ],
    )
    mutate(
        19,
        remove_positions={
            (176, 184), (696, 699), (1306, 1308), (1353, 1361),
            (1387, 1395), (1398, 1415), (1454, 1468), (1475, 1477),
            (1512, 1536), (1543, 1551), (1660, 1684), (1758, 1780),
            (1782, 1790), (1795, 1806), (2108, 2117),
        },
    )
    update_by_text(
        19,
        {
            "MÀY ĐAY MẠN TÍNH": {"candidates": ["L50.8"]},
            "mày đay vô căn": {"candidates": ["L50.1"]},
        },
    )
    mutate(
        20,
        remove_positions={(1063, 1080), (1448, 1458), (2246, 2280), (2281, 2292)},
    )
    mutate(21, remove_positions={(750, 757)}, updates={(713, 728): {"type": SYMPTOM}})
    update_by_text(
        21,
        {
            "Bệnh thoái hóa tinh bột": {"candidates": ["E85.9"]},
            "bệnh thoái hóa tinh bột": {"candidates": ["E85.9"]},
            "bệnh amyloidosis": {"candidates": ["E85.9"]},
            "amyloidosis": {"candidates": ["E85.9"]},
            "Bệnh amyloidosis tự miễn dịch": {"candidates": ["E85.3"]},
            "Bệnh amyloidosis di truyền hoặc gia đình": {"candidates": ["E85.2"]},
        },
    )
    mutate(22, remove_positions={(1205, 1220), (1442, 1445), (1458, 1466)})
    update_by_text(22, {"dị ứng thời tiết": {"candidates": ["J30.9"]}})
    mutate(25, remove_positions={(235, 248)})
    update_by_text(
        25,
        {
            "tắc ống dẫn trứng": {"candidates": ["N97.1"]},
            "ung thư cổ tử cung": {"candidates": ["C53.9"]},
            "mụn trứng cá": {"candidates": ["L70.9"]},
            "tàn nhang": {"candidates": ["L81.2"]},
        },
    )


def curate_files_26_to_50() -> None:
    mutate(
        26,
        remove_positions={
            (288, 291), (699, 711), (798, 812), (838, 841), (885, 897),
            (944, 952), (955, 968), (971, 973), (976, 984), (1061, 1064),
            (1146, 1149), (1253, 1256), (2195, 2197),
        },
        updates={(661, 674): {"candidates": []}},
        additions=[
            {"text": "đau thắt ngực", "type": DIAGNOSIS, "candidates": ["I20.9"]},
            {"text": "Đau dữ dội, kéo dài", "type": SYMPTOM},
            {"text": "Nhồi máu cơ tim ST chênh", "type": DIAGNOSIS, "candidates": ["I21.3"]},
            {"text": "Đau thắt ngực không ổn định", "type": DIAGNOSIS, "candidates": ["I20.0"]},
        ],
    )
    update_by_text(
        26,
        {
            "BỆNH MẠCH VÀNH": {"candidates": ["I25.1"]},
            "Bệnh mạch vành": {"candidates": ["I25.1"]},
            "xơ vữa động mạch": {"candidates": ["I70.9"]},
            "Xơ vữa động mạch vành": {"candidates": ["I25.1"]},
            "nhồi máu cơ tim": {"candidates": ["I21.9"]},
            "Nhồi máu cơ tim": {"candidates": ["I21.9"]},
            "Tăng huyết áp": {"candidates": ["I10"]},
            "Đái tháo đường": {"candidates": ["E11.9"]},
            "Rối loạn lipid máu": {"candidates": ["E78.5"]},
            "Béo phì": {"candidates": ["E66.9"]},
        },
    )
    mutate(
        27,
        remove_positions={(250, 258), (365, 389), (466, 496)},
        updates={
            (511, 527): {"assertions": []},
            (529, 547): {"assertions": []},
            (2201, 2243): {"assertions": []},
            (2255, 2265): {"type": DIAGNOSIS, "candidates": ["J85.2"]},
        },
        additions=[{"text": "thuốc kháng sinh", "type": DRUG}],
    )
    update_by_text(
        27,
        {
            "áp xe phổi": {"candidates": ["J85.2"]},
            "Áp xe phổi": {"candidates": ["J85.2"]},
            "áp-xe nhỏ": {"candidates": ["J85.2"]},
            "viêm phổi": {"candidates": ["J18.9"]},
            "Viêm phổi": {"candidates": ["J18.9"]},
        },
    )
    mutate(
        28,
        remove_positions={
            (176, 184), (878, 880), (897, 919), (925, 933), (959, 967),
            (972, 987), (994, 1020), (1047, 1049), (1084, 1108),
            (1115, 1128), (1257, 1280), (1330, 1352), (1354, 1362),
            (1367, 1378), (1435, 1445), (1530, 1543),
        },
        updates={(857, 871): {"type": RESULT}},
    )
    update_by_text(
        28,
        {
            "MÀY ĐAY MẠN TÍNH": {"candidates": ["L50.8"]},
            "bệnh lý mày đay vô căn": {"candidates": ["L50.1"]},
            "Mày đay vô căn": {"candidates": ["L50.1"]},
        },
    )
    mutate(29, remove_positions={(43, 45), (119, 121), (334, 342), (1823, 1850), (1935, 1948)})
    mutate(30, remove_positions={(426, 428), (719, 727)})
    update_by_text(
        30,
        {
            "MÀY đay VÔ CĂN": {"candidates": ["L50.1"]},
            "MÀY đay MẠN": {"candidates": ["L50.8"]},
            "bệnh lý mày đay vô căn": {"candidates": ["L50.1"]},
        },
    )
    mutate(31, remove_positions={(1791, 1801)})
    update_by_text(
        31,
        {
            "covid": {"candidates": ["U07.1"]},
            "Giãn thừng tinh": {"candidates": ["I86.1"]},
            "vô sinh thứ phát": {"candidates": ["N97.9"]},
        },
    )
    mutate(32, remove_positions={(750, 757)})
    update_by_text(
        32,
        {
            "amyloidosis": {"candidates": ["E85.9"]},
            "Bệnh amyloidosis chuỗi nhẹ": {"candidates": ["E85.81"]},
            "Bệnh amyloidosis tự miễn dịch": {"candidates": ["E85.3"]},
            "Bệnh amyloidosis di truyền hoặc gia đình": {"candidates": ["E85.2"]},
            "Bệnh thoái hóa tinh bột": {"candidates": ["E85.9"]},
        },
    )
    mutate(33, remove_positions={(617, 623), (662, 687), (2175, 2183)})
    mutate(
        34,
        remove_positions={
            (99, 102), (376, 379), (438, 441), (477, 480), (544, 547),
            (696, 699), (709, 712), (725, 728), (839, 842), (876, 879),
            (909, 912), (1042, 1045), (1337, 1340), (1360, 1363),
            (1503, 1506), (1783, 1786), (1934, 1937), (2112, 2124),
        },
    )
    update_by_text(
        34,
        {
            "viêm xoang": {"candidates": ["J32.9"]},
            "trầm cảm": {"candidates": ["F32.9"]},
            "đột quỵ": {"candidates": ["I63.9"]},
            "suy thoái võng mạc": {"candidates": ["H35.9"]},
        },
    )
    mutate(35, remove_positions={(935, 943), (1865, 1874)})
    update_by_text(
        35,
        {
            "Viêm hang vị sung huyết": {"candidates": ["K29.7"]},
            "viêm sung huyết hang vị dạ dày": {"candidates": ["K29.7"]},
        },
    )
    mutate(
        37,
        remove_positions={(343, 350), (468, 480), (505, 529), (632, 639), (1469, 1481), (1725, 1752), (1796, 1809)},
        updates={
            (1015, 1040): {"assertions": ["isFamily"]},
            (1494, 1510): {"assertions": ["isFamily"]},
            (1526, 1563): {"assertions": ["isFamily"]},
        },
    )
    mutate(38, remove_positions={(1774, 1792)})
    mutate(39, remove_positions={(1191, 1194)})
    update_by_text(39, {"bệnh bàn chân bẹt": {"candidates": ["M21.4"]}})
    mutate(
        40,
        remove_positions={(945, 964), (1534, 1552), (1706, 1724)},
        updates={
            (503, 518): {"type": RESULT},
            (612, 628): {"type": RESULT},
        },
    )
    mutate(
        41,
        remove_positions={(300, 306), (316, 326), (504, 516), (525, 529), (1805, 1811)},
    )
    update_by_text(
        41,
        {
            "Trứng cá": {"candidates": ["L70.0"]},
            "mụn trứng cá": {"candidates": ["L70.9"]},
        },
    )
    mutate(42, remove_positions={(51, 53), (512, 519), (903, 911), (1324, 1348)})
    mutate(43, remove_positions={(304, 307), (584, 587), (735, 738)})
    update_by_text(
        44,
        {
            "MÀY đay VÔ CĂN": {"candidates": ["L50.1"]},
            "MÀY đay MẠN": {"candidates": ["L50.8"]},
            "đay vô căn": {"candidates": ["L50.1"]},
        },
    )
    mutate(45, remove_positions={(1315, 1325)})
    mutate(
        46,
        remove_positions={(1496, 1504)},
        updates={(1022, 1106): {"type": RESULT, "assertions": []}},
    )
    mutate(47, remove_positions={(805, 812), (903, 910)})
    mutate(
        48,
        remove_positions={(343, 350), (468, 480), (632, 639), (965, 972), (1589, 1616)},
        updates={(1390, 1427): {"assertions": ["isFamily"]}},
    )
    update_by_text(
        49,
        {
            "viêm da tiếp xúc": {"candidates": ["L23.9"]},
            "psoralen": {"candidates": ["2103294"]},
        },
    )
    mutate(
        50,
        remove_positions={(1464, 1474), (1726, 1737)},
        updates={
            (1405, 1409): {"candidates": ["J85.0"]},
            (1586, 1590): {"candidates": ["J85.0"]},
            (1649, 1667): {"type": RESULT},
            (1674, 1693): {"type": RESULT, "assertions": []},
            (1759, 1788): {"type": RESULT},
        },
    )


def curate_files_51_to_75() -> None:
    mutate(51, updates={(1817, 1831): {"type": RESULT}})
    mutate(
        52,
        remove_positions={
            (176, 184), (878, 880), (925, 933), (959, 967), (970, 987),
            (994, 1020), (1047, 1049), (1115, 1123), (1129, 1158),
            (1225, 1256), (1257, 1280), (1297, 1324), (1325, 1352),
            (1354, 1362), (1367, 1378), (1759, 1766),
        },
        updates={(857, 871): {"type": RESULT}},
        additions=[
            {"text": "Nhuộm huỳnh quang miễn dịch", "type": TEST},
            {"text": "lắng đọng các phức hợp miễn dịch, bổ thể hay sợi fibrin", "type": RESULT},
        ],
    )
    update_by_text(
        52,
        {
            "MÀY ĐAY MẠN TÍNH": {"candidates": ["L50.8"]},
            "bệnh lý mày đay vô căn": {"candidates": ["L50.1"]},
            "Mày đay vô căn": {"candidates": ["L50.1"]},
        },
    )
    mutate(53, remove_positions={(896, 906), (907, 917)})
    mutate(
        54,
        remove_positions={(33, 39), (130, 148), (259, 269), (1276, 1286), (1323, 1333), (1461, 1471), (1493, 1503), (1738, 1744)},
        updates={
            (1208, 1223): {"assertions": []},
            (1688, 1695): {"assertions": ["isNegated"]},
        },
    )
    mutate(55, remove_positions={(206, 214), (427, 435), (782, 790)})
    mutate(56, remove_positions={(603, 611), (1533, 1542), (1572, 1580)})
    update_by_text(
        56,
        {
            "Viêm hang vị sung huyết": {"candidates": ["K29.7"]},
            "bệnh viêm sung huyết hang vị dạ dày": {"candidates": ["K29.7"]},
        },
    )
    mutate(57, remove_positions={(566, 576), (577, 587)})
    mutate(
        58,
        remove_positions={(731, 755), (764, 810), (896, 909), (948, 975), (1134, 1144), (1145, 1155), (1317, 1327)},
    )
    mutate(59, remove_positions={(541, 547), (668, 686), (991, 1001), (1002, 1018), (1532, 1538), (1633, 1639)})
    update_by_text(59, {"mụn trứng cá": {"candidates": ["L70.9"]}})
    mutate(
        60,
        remove_positions={(541, 547), (557, 567), (586, 596), (985, 995), (1105, 1115), (1288, 1298), (1530, 1536), (1631, 1637)},
    )
    mutate(61, remove_positions={(728, 738), (1352, 1370)})
    mutate(
        62,
        remove_positions={(643, 651)},
        updates={
            (29, 42): {"assertions": ["isFamily"]},
            (1064, 1088): {"assertions": ["isFamily"]},
        },
    )
    mutate(63, remove_positions={(482, 485), (762, 765), (782, 822), (913, 916), (1478, 1481)})
    mutate(
        64,
        remove_positions={(948, 951), (1255, 1263)},
        updates={
            (338, 345): {"assertions": ["isFamily"]},
            (356, 382): {"assertions": ["isFamily"]},
            (725, 732): {"assertions": ["isFamily"]},
            (969, 983): {"assertions": []},
            (1508, 1516): {"assertions": ["isFamily"]},
        },
        additions=[{"text": "đau thắt ngực", "type": DIAGNOSIS, "candidates": ["I20.9"]}],
    )
    mutate(65, remove_positions={(242, 248)})
    update_by_text(65, {"viêm da tiếp xúc": {"candidates": ["L23.9"]}})
    mutate(
        66,
        remove_positions={(872, 880), (914, 922), (1655, 1661)},
        updates={
            (1004, 1007): {"assertions": ["isNegated"]},
            (1097, 1100): {"assertions": ["isNegated"]},
            (1369, 1377): {"assertions": ["isNegated"]},
        },
        additions=[
            {"text": "hạ huyết áp tư thế đứng", "type": DIAGNOSIS, "candidates": ["I95.1"]},
        ],
    )
    update_by_text(
        66,
        {
            "khối u ung thư": {"candidates": ["C80.1"]},
            "khối u lành tính": {"candidates": ["D36.9"]},
            "thiếu vitamin B12": {"candidates": ["E53.8"]},
            "viêm teo niêm mạc dạ dày": {"candidates": ["K29.4"]},
            "thiếu máu": {"candidates": ["D64.9"]},
        },
    )
    mutate(67, remove_positions={(528, 536), (1458, 1467)})
    update_by_text(
        67,
        {
            "Viêm hang vị sung huyết": {"candidates": ["K29.7"]},
            "bệnh viêm sung huyết hang vị dạ dày": {"candidates": ["K29.7"]},
        },
    )
    mutate(
        68,
        updates={
            (1360, 1369): {"assertions": ["isFamily"]},
            (1498, 1510): {"assertions": ["isFamily"]},
        },
    )
    mutate(
        69,
        remove_positions={(515, 533), (1127, 1130), (1143, 1151), (1605, 1622)},
        updates={
            (260, 266): {"assertions": ["isHistorical"]},
            (1337, 1353): {"assertions": ["isNegated"]},
        },
    )
    mutate(70, remove_positions={(721, 735), (773, 796)}, updates={(1631, 1645): {"type": RESULT}})
    mutate(71, remove_positions={(66, 84), (1345, 1351)})
    update_by_text(
        71,
        {
            "suy giảm chức năng gan": {"candidates": ["K76.9"]},
            "nổi mề đay": {"candidates": ["L50.9"]},
            "bệnh vảy nến": {"candidates": ["L40.9"]},
        },
    )
    mutate(73, remove_positions={(140, 144)}, updates={(1041, 1052): {"assertions": [], "candidates": ["E85.9"]}})
    update_by_text(
        73,
        {
            "amyloidosis": {"candidates": ["E85.9"]},
            "Bệnh thoái hóa tinh bột": {"candidates": ["E85.9"]},
        },
    )
    mutate(75, remove_positions={(264, 283), (292, 309), (1498, 1504)})
    update_by_text(
        75,
        {
            "Nấm bẹn": {"candidates": ["B35.6"]},
            "béo phì": {"candidates": ["E66.9"]},
            "nhiễm nấm da": {"candidates": ["B35.9"]},
        },
    )


def curate_files_76_to_100() -> None:
    mutate(
        76,
        remove_positions={(45, 62), (981, 994)},
        updates={
            (124, 135): {"assertions": []},
            (1280, 1288): {"assertions": ["isNegated"]},
        },
    )
    update_by_text(
        76,
        {
            "MÀY đay VÔ CĂN": {"candidates": ["L50.1"]},
            "MÀY đay MẠN": {"candidates": ["L50.8"]},
            "đay vô căn": {"candidates": ["L50.1"]},
        },
    )
    mutate(77, remove_positions={(495, 498)})
    mutate(78, additions=[{"text": "nội soi mũi họng", "type": TEST}])
    update_by_text(78, {"viêm nhiễm": {"candidates": ["B99.9"]}})
    mutate(79, remove_positions={(750, 757)}, updates={(770, 799): {"type": RESULT}})
    update_by_text(
        79,
        {
            "bệnh amyloidosis": {"candidates": ["E85.9"]},
            "bệnh thoái hóa tinh bột": {"candidates": ["E85.9"]},
            "amyloidosis": {"candidates": ["E85.9"]},
            "bệnh ung thư máu": {"candidates": ["C96.9"]},
            "đa u tủy": {"candidates": ["C90.0"]},
            "Bệnh amyloidosis chuỗi nhẹ": {"candidates": ["E85.81"]},
            "Bệnh amyloidosis tự miễn dịch": {"candidates": ["E85.3"]},
        },
    )
    update_by_text(80, {"nhiễm trùng": {"candidates": ["A49.9"]}})
    mutate(81, remove_positions={(199, 201), (612, 621)})
    update_by_text(
        81,
        {
            "gen đông máu": {"candidates": ["D68.59"]},
            "bệnh tăng đông máu": {"candidates": ["D68.59"]},
            "sẩy thai": {"candidates": ["O03.9"]},
            "sinh non": {"candidates": ["O60.1"]},
            "tiền sản giật": {"candidates": ["O14.9"]},
            "tắc mạch": {"candidates": ["I74.9"]},
        },
    )
    mutate(82, remove_positions={(623, 631)})
    update_by_text(82, {"amyloidosis": {"candidates": ["E85.9"]}})
    mutate(83, updates={(1183, 1191): {"assertions": ["isNegated"]}})
    update_by_text(
        83,
        {
            "MÀY đay VÔ CĂN": {"candidates": ["L50.1"]},
            "MÀY đay MẠN": {"candidates": ["L50.8"]},
            "đay vô căn": {"candidates": ["L50.1"]},
        },
    )
    mutate(84, remove_positions={(158, 177), (186, 203), (1392, 1398)})
    update_by_text(
        84,
        {
            "Nấm bẹn": {"candidates": ["B35.6"]},
            "béo phì": {"candidates": ["E66.9"]},
            "nhiễm nấm da": {"candidates": ["B35.9"]},
        },
    )
    mutate(
        86,
        remove_positions={(192, 203), (324, 332), (386, 396), (1254, 1263)},
        updates={(592, 610): {"type": SYMPTOM}},
    )
    update_by_text(
        86,
        {
            "Viêm hang vị sung huyết": {"candidates": ["K29.7"]},
            "viêm sung huyết hang vị dạ dày": {"candidates": ["K29.7"]},
        },
    )
    mutate(87, remove_positions={(48, 57), (442, 452)})
    mutate(
        88,
        remove_positions={(837, 848), (873, 883)},
        updates={
            (921, 949): {"assertions": ["isFamily"]},
            (951, 971): {"assertions": ["isFamily"]},
            (1083, 1095): {"assertions": ["isFamily"]},
            (1308, 1320): {"assertions": ["isFamily"]},
        },
    )
    mutate(89, remove_positions={(992, 1005)})
    mutate(90, remove_positions={(61, 63)})
    mutate(
        91,
        remove_positions={(633, 636)},
        additions=[{"text": "Sỏi niệu quản", "type": DIAGNOSIS, "candidates": ["N20.1"]}],
    )
    mutate(92, remove_positions={(881, 895), (943, 953)})
    mutate(93, remove_positions={(206, 214), (427, 435), (782, 790)})
    mutate(94, remove_positions={(235, 243), (1165, 1174)})
    update_by_text(
        94,
        {
            "Viêm hang vị sung huyết": {"candidates": ["K29.7"]},
            "viêm sung huyết hang vị dạ dày": {"candidates": ["K29.7"]},
        },
    )
    mutate(
        95,
        remove_positions={(1024, 1033), (1086, 1089), (1162, 1164)},
        additions=[{"text": "đau các khớp rất nhiều", "type": SYMPTOM}],
    )
    update_by_text(95, {"cơn gout cấp": {"candidates": ["M10.9"]}})
    mutate(
        96,
        remove_positions={(780, 792), (839, 846), (853, 868)},
        updates={(592, 610): {"type": SYMPTOM}},
    )
    mutate(97, remove_positions={(439, 447)})
    mutate(98, remove_positions={(208, 215)})
    update_by_text(100, {"tiền sản giật": {"candidates": ["O14.9"]}})
    mutate(100, updates={(878, 885): {"assertions": ["isNegated"]}})


def main() -> None:
    curate_file_1()
    curate_file_2()
    curate_file_3()
    curate_file_4()
    curate_file_5()
    curate_file_6()
    curate_file_7()
    curate_file_8()
    curate_file_9()
    curate_file_10()
    curate_files_11_to_25()
    curate_files_26_to_50()
    curate_files_51_to_75()
    curate_files_76_to_100()
    normalize_candidate_ids()


if __name__ == "__main__":
    main()
