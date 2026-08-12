"""TRAKE — truy xuất video + căn chỉnh chuỗi khoảnh khắc theo thời gian.

Issue #19. Luật và cách chấm: docs/onboarding/10.

    Nộp:  <video_id>, <frame_id_1>, ..., <frame_id_N>
    Chấm: sai video -> 0 điểm ngay. Đúng video -> (số khoảnh khắc khớp) / N.
          Mỗi khoảnh khắc chỉ đúng trong đoạn [s_j, e_j] thường DƯỚI 10 frame.

Hai giai đoạn:

  1. TRUY XUẤT — dùng `video_search_en` tìm top-K video ứng viên qua index CLIP
     có sẵn (backend/stores_faiss.py). Đây là phần ăn phần lớn điểm.

  2. CĂN CHỈNH — trích frame DÀY từ video ứng viên (keyframe I-frame của BTC quá
     thưa so với cửa sổ 10 frame), chấm từng frame với từng mô tả khoảnh khắc,
     rồi chọn bộ frame TĂNG DẦN có tổng điểm cao nhất — chính là align_events().

Module này lo phần 1 (decompose) và phần chọn frame (align_events). Việc trích
frame bằng ffmpeg nằm ở indexing/extract_frames.py.
"""
import json
import os

from agent.llm import call_llm, parse_json
from agent.prompts import TRAKE_FEWSHOT, TRAKE_SYSTEM

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Giai đoạn 1 — tách truy vấn thành chuỗi khoảnh khắc
# ---------------------------------------------------------------------------

def decompose(query_vi: str) -> dict:
    """Truy vấn tiếng Việt -> {video_search_en, events:[{name, search_en}]}."""
    provider = os.environ.get("AIC_LLM_PROVIDER", "mock")
    raw = (_mock_decompose(query_vi) if provider == "mock"
           else call_llm(TRAKE_SYSTEM, TRAKE_FEWSHOT, query_vi))
    result = parse_json(raw)

    if "video_search_en" not in result or "events" not in result:
        raise ValueError(f"LLM thiếu key bắt buộc — raw: {raw[:200]}")
    if not isinstance(result["events"], list) or not result["events"]:
        raise ValueError(f"events phải là danh sách không rỗng — raw: {raw[:200]}")
    for i, ev in enumerate(result["events"]):
        if "search_en" not in ev:
            raise ValueError(f"events[{i}] thiếu search_en — raw: {raw[:200]}")
        ev.setdefault("name", f"khoảnh khắc {i + 1}")
    return result


def _mock_decompose(query_vi: str) -> str:
    """Rule-based tối giản cho dev offline. KHÔNG dùng khi thi."""
    import re
    n = 3
    m = re.search(r"(\d+)\s*(khoảnh khắc|giai đoạn|bước|sự kiện)", query_vi.lower())
    if m:
        n = int(m.group(1))
    return json.dumps({
        "video_search_en": f"(mock video search: {query_vi[:60]})",
        "events": [{"name": f"khoảnh khắc {i + 1}",
                    "search_en": f"(mock event {i + 1})"} for i in range(n)],
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Giai đoạn 2 — căn chỉnh: chọn frame tăng dần tối ưu
# ---------------------------------------------------------------------------

def align_events(scores, min_gap: int = 1):
    """Chọn cho mỗi sự kiện một khung hình, sao cho thứ tự thời gian được giữ.

    scores[j][i] = độ khớp của sự kiện j với khung hình thứ i.
        Khung hình PHẢI đã sắp xếp tăng dần theo thời gian.

    Trả về (danh sách chỉ số khung hình, tổng điểm), hoặc (None, -inf) nếu
    không đủ khung hình để xếp.

    Vì sao cần hàm này thay vì lấy argmax từng sự kiện: argmax độc lập hay cho
    ra thứ tự vô lý (tiếp đất trước khi giậm nhảy). Ràng buộc tăng dần vừa loại
    các tổ hợp bất khả thi, vừa tận dụng chính cấu trúc thời gian mà đề bài cho.

    Quy hoạch động O(N x M) với biến chạy giữ max tiền tố:
        dp[j][i] = scores[j][i] + max(dp[j-1][i'] với i' <= i - min_gap)
    """
    n_ev = len(scores)
    if n_ev == 0:
        return None, NEG_INF
    n_fr = len(scores[0])
    if n_fr < (n_ev - 1) * min_gap + 1:
        return None, NEG_INF  # không đủ khung hình để xếp đủ N sự kiện

    dp = [[NEG_INF] * n_fr for _ in range(n_ev)]
    par = [[-1] * n_fr for _ in range(n_ev)]
    dp[0] = list(scores[0])

    for j in range(1, n_ev):
        best_val, best_i = NEG_INF, -1
        for i in range(n_fr):
            cand = i - min_gap
            if cand >= 0 and dp[j - 1][cand] > best_val:
                best_val, best_i = dp[j - 1][cand], cand
            if best_i >= 0:
                dp[j][i] = scores[j][i] + best_val
                par[j][i] = best_i

    last = max(range(n_fr), key=lambda i: dp[n_ev - 1][i])
    total = dp[n_ev - 1][last]
    if total == NEG_INF:
        return None, NEG_INF

    path = [last]
    for j in range(n_ev - 1, 0, -1):
        last = par[j][last]
        path.append(last)
    path.reverse()
    return path, total


def rank_videos(hits, get_metadata, top_videos: int = 10,
                thuong_nhieu_hit: float = 0.02):
    """Giai đoạn 1: gộp kết quả CLIP theo VIDEO, xếp hạng video ứng viên.

    hits: [(keyframe_id, score)] trả về từ store.search_text(video_search_en).

    Vì sao gộp theo video chứ không lấy thẳng top keyframe: TRAKE chấm **sai
    video là 0 điểm ngay**, nên việc chọn đúng video quan trọng hơn hẳn việc
    chọn đúng frame. Một truy vấn TRAKE mô tả cả chuỗi sự kiện trải dài theo
    thời gian, nên video đúng thường có NHIỀU keyframe cùng khớp — đó là tín
    hiệu thật, không phải trùng hợp. `thuong_nhieu_hit` thưởng nhẹ cho điều đó,
    đủ nhỏ để không lấn át một video khớp mạnh ở một chỗ.

    Trả về [{video_id, center_frame, best_keyframe_id, score, n_hits}] —
    `center_frame` là tâm để giai đoạn 2 quét frame dày xung quanh.
    """
    import math

    theo_video: dict[str, dict] = {}
    for kf_id, score in hits:
        meta = get_metadata(kf_id)
        if not meta:
            continue
        v = meta["video_id"]
        cur = theo_video.get(v)
        if cur is None:
            theo_video[v] = {"video_id": v, "score": score, "n_hits": 1,
                             "best_keyframe_id": kf_id,
                             "center_frame": meta.get("frame_idx", 0)}
        else:
            cur["n_hits"] += 1
            if score > cur["score"]:
                cur["score"] = score
                cur["best_keyframe_id"] = kf_id
                cur["center_frame"] = meta.get("frame_idx", 0)

    ds = list(theo_video.values())
    for v in ds:
        v["score_cuoi"] = v["score"] + thuong_nhieu_hit * math.log1p(v["n_hits"])
    ds.sort(key=lambda v: -v["score_cuoi"])
    return ds[:top_videos]


def solve_alignment(video_path, center_frame: int, events: list[dict], encoder,
                    radius: int = 150, step: int = 3, min_gap: int = 1,
                    size=(224, 224)):
    """Giai đoạn 2 trọn vẹn: từ video ứng viên -> N frame_id để nộp.

    video_path    : file video gốc (BTC nói đây mới là dữ liệu thi chính thức)
    center_frame  : vị trí thô do CLIP chỉ ra ở giai đoạn 1 (frame_idx của keyframe)
    events        : [{name, search_en}] từ decompose()
    encoder       : agent.clip_encoder.ClipEncoder
    radius, step  : quét ±radius frame quanh tâm, lấy 1 frame mỗi `step` frame.
                    Đoạn đáp án thường dưới 10 frame nên step <= 5 mới an toàn.

    Trả về (danh sách frame_id, tổng điểm, ma trận điểm) — frame_id là con số
    nộp cho BTC, KHÔNG phải vị trí trong danh sách ứng viên.
    """
    from indexing.extract_frames import extract_window

    frame_ids, imgs = extract_window(video_path, center_frame, radius, step, size)
    if not frame_ids:
        return None, NEG_INF, None

    V_img = encoder.encode_images(imgs)                        # (M, D)
    V_txt = encoder.encode_text([ev["search_en"] for ev in events])  # (N, D)
    scores = (V_txt @ V_img.T).tolist()                         # (N, M)

    chosen, total = pick_frames(scores, frame_ids, min_gap)
    return chosen, total, scores


def pick_frames(events_scores, frame_ids, min_gap: int = 1):
    """Bọc align_events: nhận điểm + danh sách frame_id thật, trả frame_id đã chọn.

    frame_ids[i] = chỉ số khung hình trong video gốc của ứng viên thứ i
                   (đây mới là con số nộp cho BTC, KHÔNG phải vị trí i).
    """
    path, total = align_events(events_scores, min_gap)
    if path is None:
        return None, NEG_INF
    return [frame_ids[i] for i in path], total


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or (
        "Tìm 4 khoảnh khắc chính khi vận động viên nhảy cao: chạy đà, "
        "giậm nhảy, bay qua xà, tiếp đất.")
    print(json.dumps(decompose(q), ensure_ascii=False, indent=2))
