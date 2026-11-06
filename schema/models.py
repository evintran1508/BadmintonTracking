from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class VideoMeta(BaseModel):
    fps: float = Field(gt=0)
    total_frames: int = Field(ge=1)
    duration_sec: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CourtDims(BaseModel):
    width: float = Field(gt=0)
    length: float = Field(gt=0)
    net_y: float = Field(gt=0)


class CalibPoint(BaseModel):
    px: Tuple[float, float]
    court_m: Tuple[float, float]


class CourtMeta(BaseModel):
    mode: Literal["singles", "doubles"]
    dims_m: CourtDims
    calib_points: List[CalibPoint]
    homography_img_to_court: List[float] = Field(min_length=9, max_length=9)
    reproj_error_m: float = Field(ge=0)


class PlayerInfo(BaseModel):
    id: str
    name: str
    team: Literal["A", "B"]
    color: str


class Rally(BaseModel):
    id: int
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    shot_count: int = Field(ge=0)
    winner_team: Optional[Literal["A", "B"]] = None
    score_after: Optional[Tuple[int, int]] = None


class Shot(BaseModel):
    idx: int = Field(ge=1)
    rally_id: int
    frame: int = Field(ge=0)
    player_id: str
    shot_type: Optional[Literal["Smash", "Clear", "Drop", "Drive", "Serve", "Lift"]] = None
    speed_kmh: Optional[float] = None
    speed_confidence: float = Field(ge=0.0, le=1.0)
    from_court_m: Tuple[float, float]
    to_court_m: Tuple[float, float]


class ShuttleFrame(BaseModel):
    x_px: Optional[float] = None
    y_px: Optional[float] = None
    visible: bool
    conf: float = Field(ge=0.0, le=1.0)


class PlayerFrame(BaseModel):
    bbox_px: Optional[Tuple[float, float, float, float]] = None
    foot_px: Optional[Tuple[float, float]] = None
    foot_court_m: Optional[Tuple[float, float]] = None
    speed_mps: float = Field(ge=0.0)
    dist_cum_m: float = Field(ge=0.0)
    energy: float = Field(ge=0.0, le=100.0)
    conf: float = Field(ge=0.0, le=1.0)


class FrameRecord(BaseModel):
    i: int = Field(ge=0)
    t: float = Field(ge=0.0)
    shuttle: ShuttleFrame
    players: Dict[str, PlayerFrame]


class ZonesPct(BaseModel):
    front: float = Field(ge=0.0, le=100.0)
    mid: float = Field(ge=0.0, le=100.0)
    back: float = Field(ge=0.0, le=100.0)


class PlayerSummary(BaseModel):
    total_distance_m: float = Field(ge=0.0)
    avg_speed_mps: float = Field(ge=0.0)
    max_speed_mps: float = Field(ge=0.0)
    zones_pct: ZonesPct


class MatchDocument(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    match_id: str
    video: VideoMeta
    court: CourtMeta
    players: List[PlayerInfo]
    rallies: List[Rally]
    shots: List[Shot]
    frames: List[FrameRecord]
    summary: Dict[str, PlayerSummary]

    @model_validator(mode="after")
    def check_monotonic_invariants(self) -> "MatchDocument":
        if len(self.frames) != self.video.total_frames:
            raise ValueError(f"Expected {self.video.total_frames} frames, got {len(self.frames)}")
        prev_dist: Dict[str, float] = {}
        prev_energy: Dict[str, float] = {}
        for idx, frame in enumerate(self.frames):
            if frame.i != idx:
                raise ValueError(f"Frame index mismatch at pos {idx}: got i={frame.i}")
            for pid, pf in frame.players.items():
                if pid in prev_dist and pf.dist_cum_m < prev_dist[pid] - 1e-6:
                    raise ValueError(
                        f"dist_cum_m decreased for player {pid} at frame {frame.i}: "
                        f"{pf.dist_cum_m} < {prev_dist[pid]}"
                    )
                if pid in prev_energy and pf.energy > prev_energy[pid] + 1e-6:
                    raise ValueError(
                        f"energy increased for player {pid} at frame {frame.i}: "
                        f"{pf.energy} > {prev_energy[pid]}"
                    )
                prev_dist[pid] = pf.dist_cum_m
                prev_energy[pid] = pf.energy
        return self

