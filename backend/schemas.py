"""Pydantic models — hợp đồng API v1 (docs/onboarding/00, mục 3).

Chủ sở hữu: Tiến Khang (N2). Đổi schema phải báo cả đội.
"""
from typing import Optional

from pydantic import BaseModel, Field


class Filters(BaseModel):
    video_ids: Optional[list[str]] = None
    publish_date_from: Optional[str] = None
    publish_date_to: Optional[str] = None
    objects: Optional[list[str]] = None
    channel: Optional[str] = None


class SearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_variants: list[str] = Field(default_factory=list)
    query_image_b64: Optional[str] = None
    filters: Filters = Field(default_factory=Filters)
    top_k: int = 100
    # Vòng sơ tuyển chỉ có kis | qa | trake (docs/onboarding/10).
    # "avs" giữ lại cho vòng sau, hiện không dùng.
    mode: str = "kis"


class FeedbackRequest(BaseModel):
    positive_keyframe_ids: list[str] = Field(default_factory=list)
    negative_keyframe_ids: list[str] = Field(default_factory=list)
    query_text: Optional[str] = None
    top_k: int = 100


class SearchResult(BaseModel):
    video_id: str
    keyframe_id: str
    # frame_idx = CON SỐ NỘP CHO BTC (chỉ số khung hình trong video gốc).
    # BTC chấm đúng/sai bằng `frame_idx ∈ [s, e]`, KHÔNG dùng giây.
    # start_time/end_time chỉ để UI tua video. Xem docs/onboarding/10.
    frame_idx: int
    start_time: float
    end_time: float
    score: float
    signals: dict = Field(default_factory=dict)
    reason: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    latency_ms: float


class TimelineEntry(BaseModel):
    keyframe_id: str
    frame_idx: int
    timestamp: float


class TimelineResponse(BaseModel):
    video_id: str
    entries: list[TimelineEntry]
