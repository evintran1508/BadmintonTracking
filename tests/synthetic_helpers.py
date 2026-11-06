"""Synthetic color-filter helpers ONLY for unit tests on `synthetic_*` fixtures.

Per Plan v3 & AGENTS.md Rule 8/9:
- NEVER import or call these functions inside `ai_core/` or production code.
- Synthetic metrics do NOT count toward acceptance gates.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def detect_synthetic_players(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    b, g, r = cv2.split(frame_bgr.astype(np.int16))
    is_outer = (b < 55) & (g < 65) & (r < 55)
    is_court = (np.abs(b - 68) < 30) & (np.abs(g - 125) < 30) & (np.abs(r - 52) < 30)
    mask = ((~(is_outer | is_court)).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(opened)

    dets: List[Dict[str, Any]] = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area < 160 or h < 24 or w > 90:
            continue
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        cx = x + w * 0.5
        by = y + h
        kps = [[0.0, 0.0, 0.0]] * 15 + [[cx - 4.0, by - 1.0, 0.92], [cx + 4.0, by - 1.0, 0.92]]
        dets.append({"raw_track_id": lbl, "bbox_px": bbox, "keypoints": kps, "conf": 0.93})
    return dets


def detect_synthetic_shuttle(
    frame_bgr: np.ndarray,
    prev_bgr: Optional[np.ndarray] = None,
    *_args: Any,
    **_kwargs: Any,
) -> Dict[str, Any]:
    b, g, r = cv2.split(frame_bgr.astype(np.int16))
    yellow_mask = ((r > 220) & (g > 220) & (b < 80)).astype(np.uint8) * 255
    if prev_bgr is not None:
        diff_y = cv2.absdiff(frame_bgr, prev_bgr)
        motion_y = (np.max(diff_y, axis=2) > 25).astype(np.uint8) * 255
        yellow_mask = cv2.bitwise_and(yellow_mask, motion_y)

    num_y, _ly, stats_y, cent_y = cv2.connectedComponentsWithStats(yellow_mask)
    best_y: Optional[Tuple[float, float]] = None
    best_area_y = 0
    for lbl in range(1, num_y):
        area = int(stats_y[lbl, cv2.CC_STAT_AREA])
        if 6 <= area <= 180 and area > best_area_y:
            best_area_y = area
            best_y = (round(float(cent_y[lbl][0]), 2), round(float(cent_y[lbl][1]), 2))
    if best_y is not None:
        return {"x_px": best_y[0], "y_px": best_y[1], "visible": True, "conf": 0.90}
    return {"x_px": None, "y_px": None, "visible": False, "conf": 0.0}

