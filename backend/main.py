"""AIC 2026 — Backend lõi truy hồi (N2 - Tiến Khang).

Chạy:  uvicorn backend.main:app --reload --port 8000
UI:    http://localhost:8000  (serve ui/index.html)

Tuần 1 (M1): chạy trên MockStore. Khi có dữ liệu thật (25/07):
thay `from backend.stores_mock import MockStore` bằng store thật của N1
(cùng chữ ký hàm) — các endpoint giữ nguyên.
"""
import base64
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import (FeedbackRequest, SearchRequest, SearchResponse,
                             SearchResult, TimelineEntry, TimelineResponse)
from backend.stores_mock import THUMB_DIR, MockStore, generate_thumbnails

SEGMENT_GAP_SEC = 10.0  # keyframe cùng video cách nhau <= ngưỡng này thì gộp 1 segment
INDEX_VERSION = "v0_mock"

app = FastAPI(title="AIC2026 Retrieval API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

store = MockStore()
generate_thumbnails(store)


def group_segments(hits: list[tuple[str, float]], mode: str) -> list[SearchResult]:
    """Gộp keyframe liền kề cùng video thành segment (docs/onboarding/02, mục 2.2)."""
    enriched = []
    for kf_id, score in hits:
        meta = store.get_metadata(kf_id)
        if meta:
            enriched.append((meta, score))

    by_video: dict[str, list] = {}
    for meta, score in enriched:
        by_video.setdefault(meta["video_id"], []).append((meta, score))

    segments = []
    for video_id, items in by_video.items():
        items.sort(key=lambda x: x[0]["timestamp"])
        cur = None
        for meta, score in items:
            t = meta["timestamp"]
            if cur is None or t - cur["end"] > SEGMENT_GAP_SEC:
                if cur:
                    segments.append(cur)
                cur = {"video_id": video_id, "start": t, "end": t,
                       "score": score, "best_kf": meta["keyframe_id"], "count": 1}
            else:
                cur["end"] = t
                cur["count"] += 1
                if score > cur["score"]:
                    cur["score"] = score
                    cur["best_kf"] = meta["keyframe_id"]
        if cur:
            segments.append(cur)

    # AVS: khử trùng lặp trong video — giữ tối đa 3 segment/video (bản mock đơn giản)
    if mode == "avs":
        seen: dict[str, int] = {}
        deduped = []
        for s in sorted(segments, key=lambda s: -s["score"]):
            if seen.get(s["video_id"], 0) < 3:
                seen[s["video_id"]] = seen.get(s["video_id"], 0) + 1
                deduped.append(s)
        segments = deduped

    segments.sort(key=lambda s: -s["score"])
    return [SearchResult(video_id=s["video_id"], keyframe_id=s["best_kf"],
                         start_time=round(max(0.0, s["start"] - 2), 1),
                         end_time=round(s["end"] + 2, 1),
                         score=round(s["score"], 4),
                         signals={"n_keyframes": s["count"]})
            for s in segments]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.perf_counter()
    filter_ids = None
    if req.filters.video_ids:
        filter_ids = [k["keyframe_id"] for k in store.keyframes
                      if k["video_id"] in set(req.filters.video_ids)]

    if req.query_image_b64:
        img = base64.b64decode(req.query_image_b64)
        hits = store.search_image(img, top_k=req.top_k * 3, filter_ids=filter_ids)
    elif req.query_text:
        hits = store.search_text(req.query_text, top_k=req.top_k * 3,
                                 filter_ids=filter_ids)
        for variant in req.query_variants:  # RRF thật ở M2; mock: cộng dồn đơn giản
            hits += store.search_text(variant, top_k=req.top_k, filter_ids=filter_ids)
        hits = sorted({k: s for k, s in reversed(hits)}.items(), key=lambda x: -x[1])
    else:
        raise HTTPException(422, "Cần query_text hoặc query_image_b64")

    results = group_segments(hits, req.mode)[:req.top_k]
    return SearchResponse(results=results,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.post("/search/feedback", response_model=SearchResponse)
def feedback(req: FeedbackRequest):
    """Average query expansion — bản mock: ưu tiên video chứa keyframe được đánh dấu ✓."""
    t0 = time.perf_counter()
    pos_videos = {store.get_metadata(k)["video_id"]
                  for k in req.positive_keyframe_ids if store.get_metadata(k)}
    hits = store.search_text(req.query_text or "", top_k=req.top_k * 3)
    boosted = [(k, s + (1.0 if store.get_metadata(k)["video_id"] in pos_videos else 0))
               for k, s in hits if k not in set(req.negative_keyframe_ids)]
    boosted.sort(key=lambda x: -x[1])
    results = group_segments(boosted, "kis")[:req.top_k]
    return SearchResponse(results=results,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.get("/video/{video_id}/timeline", response_model=TimelineResponse)
def timeline(video_id: str, around: str | None = None, window: int = 30):
    kfs = store.timeline(video_id, around, window)
    if not kfs:
        raise HTTPException(404, f"Không có video {video_id}")
    return TimelineResponse(video_id=video_id,
                            entries=[TimelineEntry(keyframe_id=k["keyframe_id"],
                                                   timestamp=k["timestamp"])
                                     for k in kfs])


@app.get("/keyframe/{keyframe_id}/image")
def keyframe_image(keyframe_id: str, size: str = "thumb"):
    path = THUMB_DIR / f"{keyframe_id.replace('#', '_')}.jpg"
    if not path.exists():
        raise HTTPException(404, f"Không có ảnh {keyframe_id}")
    return FileResponse(path, media_type="image/jpeg")


# --- Agent endpoints: N3 implement ở M3-M4, tạm stub 501 ---

@app.post("/agent/query")
def agent_query():
    return JSONResponse(status_code=501,
                        content={"detail": "N3 implement ở mốc M4 (issue #6)"})


@app.post("/agent/vqa")
def agent_vqa():
    return JSONResponse(status_code=501,
                        content={"detail": "N3 implement ở mốc M4"})


# --- Vận hành (N5 dùng) ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {"index_version": INDEX_VERSION, "n_keyframes": len(store.keyframes),
            "n_videos": len({k['video_id'] for k in store.keyframes})}


# --- Serve UI (N4) ---

UI_FILE = Path(__file__).parent.parent / "ui" / "index.html"


@app.get("/")
def ui_root():
    if UI_FILE.exists():
        return FileResponse(UI_FILE, media_type="text/html")
    raise HTTPException(404, "ui/index.html chưa tồn tại")
