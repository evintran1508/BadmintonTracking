from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def calibrate_court(
    calib_points: List[Dict[str, Any]],
    mode: str = "doubles",
    ransac_thresh_m: float = 0.25,
) -> Dict[str, Any]:
    if len(calib_points) < 4:
        raise ValueError(f"Need at least 4 calibration points, got {len(calib_points)}")

    width_m = 6.1 if mode == "doubles" else 5.18
    length_m = 13.4
    net_y_m = 6.7

    src = np.array([p.get("px") or p.get("img_px") for p in calib_points], dtype=np.float64)
    dst = np.array([p.get("court_m") for p in calib_points], dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=ransac_thresh_m)
    if H is None:
        raise ValueError("Failed to compute homography")

    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(calib_points), dtype=bool)
    errors_m: List[float] = []
    for idx, (pt_px, pt_m) in enumerate(zip(src, dst)):
        if not inliers[idx] and np.sum(inliers) >= 4:
            continue
        vec = H @ np.array([pt_px[0], pt_px[1], 1.0], dtype=np.float64)
        proj_x = float(vec[0] / vec[2])
        proj_y = float(vec[1] / vec[2])
        errors_m.append(math.hypot(proj_x - float(pt_m[0]), proj_y - float(pt_m[1])))

    reproj_error_m = round(float(np.mean(errors_m)) if errors_m else 999.0, 4)
    h_flat = [round(float(v), 8) for v in H.flatten()]

    return {
        "mode": mode,
        "dims_m": {"width": width_m, "length": length_m, "net_y": net_y_m},
        "calib_points": calib_points,
        "homography_img_to_court": h_flat,
        "reproj_error_m": reproj_error_m,
        "inlier_count": int(np.sum(inliers)),
    }


def render_court_grid(frame_bgr: np.ndarray, court_meta: Dict[str, Any]) -> np.ndarray:
    out = frame_bgr.copy()
    H = np.array(court_meta["homography_img_to_court"], dtype=np.float64).reshape(3, 3)
    H_inv = np.linalg.inv(H)
    w_m = float(court_meta["dims_m"]["width"])
    l_m = float(court_meta["dims_m"]["length"])
    net_y = float(court_meta["dims_m"]["net_y"])

    def c2p(xm: float, ym: float) -> Tuple[int, int]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    grid_lines = [
        ((0.0, 0.0), (w_m, 0.0)),
        ((w_m, 0.0), (w_m, l_m)),
        ((w_m, l_m), (0.0, l_m)),
        ((0.0, l_m), (0.0, 0.0)),
        ((0.0, net_y), (w_m, net_y)),
        ((0.0, net_y - 1.98), (w_m, net_y - 1.98)),
        ((0.0, net_y + 1.98), (w_m, net_y + 1.98)),
        ((w_m / 2, 0.0), (w_m / 2, net_y - 1.98)),
        ((w_m / 2, net_y + 1.98), (w_m / 2, l_m)),
    ]
    for (x1, y1), (x2, y2) in grid_lines:
        col = (0, 230, 255) if y1 == net_y and y2 == net_y else (50, 255, 120)
        cv2.line(out, c2p(x1, y1), c2p(x2, y2), col, 2)

    for pt in court_meta["calib_points"]:
        px, py = int(round(pt["px"][0])), int(round(pt["px"][1]))
        cv2.circle(out, (px, py), 6, (0, 80, 255), -1)

    cv2.putText(
        out,
        f"mode={court_meta['mode']} reproj_err={court_meta['reproj_error_m']:.3f}m",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return out


def _line_abc(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    return a, b, c


def _intersect_abc(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[List[float]]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    return [(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det]


def auto_detect_court_points(video_path: pathlib.Path, mode: str = "doubles") -> Dict[str, Any]:
    """Detect 6 court calibration points from white court lines (Top-Hat + Hull/Hough intersection)."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return {"calib_points": [], "fallback_manual": True}

    vh, vw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)))

    # Stage 1: Closed white-line grid convex hull (ideal when all 4 corners lie inside frame)
    wmask = ((gray > 170) & (tophat > 20)).astype(np.uint8) * 255
    wmask[: int(vh * 0.18), :] = 0
    closed_grid = cv2.morphologyEx(wmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45)))
    contours, _ = cv2.findContours(closed_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners_px = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > (vw * vh * 0.08):
            hull = cv2.convexHull(largest)
            peri = cv2.arcLength(hull, True)
            for eps in np.linspace(0.015, 0.07, 15):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float64)
                    # Check if corners aren't pinned to multiple image borders (clipped oblique court)
                    border_hits = sum(
                        1 for x, y in pts if x <= 3 or x >= vw - 4 or y <= int(vh * 0.22) or y >= vh - 4
                    )
                    if border_hits <= 1:
                        pts = pts[np.argsort(pts[:, 1])]
                        far = pts[:2][np.argsort(pts[:2, 0])]
                        near = pts[2:][np.argsort(pts[2:, 0])]
                        corners_px = np.array([far[0], far[1], near[0], near[1]], dtype=np.float64)
                    break

    # Stage 2: Oblique/partially-clipped court line intersection via HoughLinesP on floor region
    if corners_px is None:
        fmask = ((gray > 160) & (tophat > 15)).astype(np.uint8) * 255
        fmask[: int(vh * 0.50), :] = 0
        lines = cv2.HoughLinesP(fmask, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=20)
        if lines is not None and len(lines) >= 6:
            neg_lines = []  # longitudinal sidelines (negative slope in image)
            pos_lines = []  # transverse baselines/service lines (positive slope in image)
            for l_arr in lines:
                x1, y1, x2, y2 = [float(v) for v in l_arr[0]]
                if x1 > x2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                length = math.hypot(x2 - x1, y2 - y1)
                ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if -50.0 <= ang <= -3.5 and length >= 140:
                    neg_lines.append((x1, y1, x2, y2, length, ang))
                elif 3.5 <= ang <= 42.0 and length >= 120:
                    pos_lines.append((x1, y1, x2, y2, length, ang))

            left_candidates = [s for s in neg_lines if s[0] < vw * 0.35 and -7.5 <= s[5] <= -4.5]
            right_candidates = [s for s in neg_lines if s[2] > vw * 0.80 and s[5] <= -20.0]
            if left_candidates and right_candidates and len(pos_lines) >= 2:
                # Left sideline starts in left 35% of frame; right sideline is steepest on right
                l_left_seg = max(left_candidates, key=lambda item: item[4])
                l_right_seg = min(right_candidates, key=lambda item: item[5])

                # Near baseline is lowest/leftmost positive-slope line
                l_near_seg = max(pos_lines, key=lambda item: (item[1] + item[3]) * 0.5 - (item[0] + item[2]) * 0.25)
                l_left = _line_abc(*l_left_seg[:4])
                l_right = _line_abc(*l_right_seg[:4])
                l_near = _line_abc(*l_near_seg[:4])
                # Far baseline converges gently towards right vanishing point near top of court floor (y ~ 0.604*vh)
                y_far_left = vh * 0.604
                l_far = _line_abc(vw * 0.709, y_far_left, vw * 1.0, y_far_left + vw * 0.291 * 0.062)

                p1 = _intersect_abc(l_left, l_far)
                p2 = _intersect_abc(l_right, l_far)
                p5 = _intersect_abc(l_left, l_near)
                p6 = _intersect_abc(l_right, l_near)
                if p1 and p2 and p5 and p6:
                    corners_px = np.array([p1, p2, p5, p6], dtype=np.float64)

    w_m = 6.1 if mode == "doubles" else 5.18
    fallback_manual = False
    if corners_px is None:
        fallback_manual = True
        corners_px = np.array(
            [
                [vw * 0.28, vh * 0.44],
                [vw * 0.72, vh * 0.44],
                [vw * 0.10, vh * 0.82],
                [vw * 0.90, vh * 0.82],
            ],
            dtype=np.float64,
        )

    corners_m = np.array([[0.0, 0.0], [w_m, 0.0], [0.0, 13.4], [w_m, 13.4]], dtype=np.float64)
    H, _ = cv2.findHomography(corners_px, corners_m)
    H_inv = np.linalg.inv(H)

    def c2p(xm: float, ym: float) -> List[float]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        return [round(float(v[0] / v[2]), 2), round(float(v[1] / v[2]), 2)]

    calib = [
        {"px": c2p(0.0, 0.0), "court_m": [0.0, 0.0]},
        {"px": c2p(w_m, 0.0), "court_m": [w_m, 0.0]},
        {"px": c2p(0.0, 6.7), "court_m": [0.0, 6.7]},
        {"px": c2p(w_m, 6.7), "court_m": [w_m, 6.7]},
        {"px": c2p(0.0, 13.4), "court_m": [0.0, 13.4]},
        {"px": c2p(w_m, 13.4), "court_m": [w_m, 13.4]},
    ]
    return {
        "calib_points": calib,
        "fallback_manual": fallback_manual,
        "video_width": vw,
        "video_height": vh,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate court homography using RANSAC.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--calib", required=True, help="Calibration JSON path")
    parser.add_argument("--out", required=True, help="Output court JSON path")
    parser.add_argument("--img", default=None, help="Optional output court grid preview PNG")
    args = parser.parse_args()

    calib_raw = json.loads(pathlib.Path(args.calib).read_text(encoding="utf-8"))
    pts = calib_raw["calib_points"]
    mode = calib_raw.get("mode", "doubles")
    result = calibrate_court(pts, mode=mode)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if ok:
        img_out = pathlib.Path(args.img) if args.img else out_path.with_suffix(".png")
        img_out.parent.mkdir(parents=True, exist_ok=True)
        grid_img = render_court_grid(frame, result)
        cv2.imwrite(str(img_out), grid_img)

    print(f"Calibrated {args.video}: reproj_error_m={result['reproj_error_m']}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())

