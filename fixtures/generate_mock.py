from __future__ import annotations
import json
import math
import pathlib
import random
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def _build_homography() -> Tuple[List[Dict[str, Any]], List[float], np.ndarray]:
    # 4 outer corners define exact perspective plane; 2 net posts derived from exact H
    corners_px = np.array([[230.0, 760.0], [656.0, 760.0], [70.0, 1460.0], [816.0, 1460.0]], dtype=np.float64)
    corners_m = np.array([[0.0, 0.0], [6.1, 0.0], [0.0, 13.4], [6.1, 13.4]], dtype=np.float64)
    H, _ = cv2.findHomography(corners_px, corners_m)
    H_inv = np.linalg.inv(H)
    net_l = _court_to_px(H_inv, 0.0, 6.7)
    net_r = _court_to_px(H_inv, 6.1, 6.7)
    calib = [
        {"px": [230.0, 760.0], "court_m": [0.0, 0.0]},
        {"px": [656.0, 760.0], "court_m": [6.1, 0.0]},
        {"px": [net_l[0], net_l[1]], "court_m": [0.0, 6.7]},
        {"px": [net_r[0], net_r[1]], "court_m": [6.1, 6.7]},
        {"px": [70.0, 1460.0], "court_m": [0.0, 13.4]},
        {"px": [816.0, 1460.0], "court_m": [6.1, 13.4]},
    ]
    h_flat = [round(float(x), 8) for x in H.flatten()]
    return calib, h_flat, H_inv


def _court_to_px(H_inv: np.ndarray, x_m: float, y_m: float) -> Tuple[float, float]:
    vec = H_inv @ np.array([x_m, y_m, 1.0], dtype=np.float64)
    return round(float(vec[0] / vec[2]), 2), round(float(vec[1] / vec[2]), 2)


def generate_mock_dict(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    fps = 30
    duration_sec = 15.0
    total_frames = int(fps * duration_sec)
    width, height = 886, 1920

    calib, h_flat, H_inv = _build_homography()

    players_meta = [
        {"id": "p1", "name": "Foam", "team": "A", "color": "#22d3ee"},
        {"id": "p2", "name": "Kotty", "team": "A", "color": "#facc15"},
        {"id": "p3", "name": "Opp 1", "team": "B", "color": "#a78bfa"},
        {"id": "p4", "name": "Opp 2", "team": "B", "color": "#4ade80"},
    ]

    # Base anchors in court meters: Team A (near, Y > 6.7), Team B (far, Y < 6.7)
    anchors = {
        "p1": (1.6, 10.4),
        "p2": (4.5, 10.8),
        "p3": (1.6, 3.0),
        "p4": (4.5, 2.6),
    }

    # 12 shots between frame 20 and frame 410
    shot_frames = [24, 58, 92, 126, 160, 194, 228, 262, 296, 330, 364, 398]
    shot_Order = ["p2", "p3", "p1", "p4", "p2", "p4", "p1", "p3", "p2", "p3", "p1", "p4"]
    shot_speeds = [214.0, 86.0, 24.0, 91.0, 57.0, 49.0, 30.0, 86.0, 59.0, 51.0, 112.0, 95.0]

    # Simulate smooth player positions per frame
    pos_history: Dict[str, List[Tuple[float, float]]] = {p["id"]: [] for p in players_meta}
    for i in range(total_frames):
        t = i / fps
        for idx_p, p in enumerate(players_meta):
            pid = p["id"]
            ax, ay = anchors[pid]
            phase = idx_p * 1.3
            dx = 0.65 * math.sin(2 * math.pi * (0.22 * t) + phase) + rng.uniform(-0.01, 0.01)
            dy = 0.85 * math.cos(2 * math.pi * (0.18 * t) + phase) + rng.uniform(-0.01, 0.01)
            cx = max(0.3, min(5.8, ax + dx))
            cy = max(0.5, min(6.2, ay + dy)) if p["team"] == "B" else max(7.2, min(12.9, ay + dy))
            pos_history[pid].append((round(cx, 3), round(cy, 3)))

    shots: List[Dict[str, Any]] = []
    for s_idx, f_num in enumerate(shot_frames):
        pid = shot_Order[s_idx]
        next_pid = shot_Order[(s_idx + 1) % len(shot_Order)]
        next_f = shot_frames[s_idx + 1] if s_idx + 1 < len(shot_frames) else 430
        fx, fy = pos_history[pid][f_num]
        tx, ty = pos_history[next_pid][next_f]
        shots.append(
            {
                "idx": s_idx + 1,
                "rally_id": 1,
                "frame": f_num,
                "player_id": pid,
                "speed_kmh": shot_speeds[s_idx],
                "speed_confidence": round(0.75 + 0.2 * ((s_idx % 3) / 2), 2),
                "from_court_m": [round(fx, 2), round(fy, 2)],
                "to_court_m": [round(tx, 2), round(ty, 2)],
            }
        )

    rallies = [{"id": 1, "start_frame": 20, "end_frame": 430, "shot_count": 12}]

    dist_cum = {p["id"]: 0.0 for p in players_meta}
    energy_cur = {p["id"]: 96.0 for p in players_meta}
    speeds_log: Dict[str, List[float]] = {p["id"]: [] for p in players_meta}
    zone_counts: Dict[str, Dict[str, int]] = {
        p["id"]: {"front": 0, "mid": 0, "back": 0} for p in players_meta
    }

    shot_at_frame = {s["frame"]: s for s in shots}
    frames: List[Dict[str, Any]] = []

    for i in range(total_frames):
        t = round(i / fps, 4)
        # Shuttle trajectory interpolation between shots
        if 20 <= i <= 430:
            active_shot = shots[0]
            next_frame = 430
            for s_idx, s in enumerate(shots):
                if i >= s["frame"]:
                    active_shot = s
                    next_frame = shots[s_idx + 1]["frame"] if s_idx + 1 < len(shots) else 430
            span = max(1, next_frame - active_shot["frame"])
            alpha = max(0.0, min(1.0, (i - active_shot["frame"]) / span))
            sx_m = active_shot["from_court_m"][0] * (1 - alpha) + active_shot["to_court_m"][0] * alpha
            sy_m = active_shot["from_court_m"][1] * (1 - alpha) + active_shot["to_court_m"][1] * alpha
            sx_px, sy_px = _court_to_px(H_inv, sx_m, sy_m)
            # Arc height in pixels during flight
            arc_lift = 140.0 * 4.0 * alpha * (1.0 - alpha)
            shuttle = {
                "x_px": round(sx_px, 1),
                "y_px": round(sy_px - arc_lift, 1),
                "visible": True,
                "conf": 0.88,
            }
        else:
            shuttle = {"x_px": None, "y_px": None, "visible": False, "conf": 0.0}

        frame_players: Dict[str, Any] = {}
        for p in players_meta:
            pid = p["id"]
            cx, cy = pos_history[pid][i]
            if i > 0:
                px_m, py_m = pos_history[pid][i - 1]
                step_m = math.hypot(cx - px_m, cy - py_m)
            else:
                step_m = 0.0
            spd = round(min(9.5, step_m * fps), 2)
            dist_cum[pid] = round(dist_cum[pid] + step_m, 3)
            shot_penalty = 0.8 if (i in shot_at_frame and shot_at_frame[i]["player_id"] == pid) else 0.0
            energy_cur[pid] = round(max(0.0, energy_cur[pid] - step_m * 1.15 - shot_penalty), 1)
            speeds_log[pid].append(spd)

            dist_net = abs(cy - 6.7)
            if dist_net <= 1.98:
                zone_counts[pid]["front"] += 1
            elif dist_net <= 4.72:
                zone_counts[pid]["mid"] += 1
            else:
                zone_counts[pid]["back"] += 1

            foot_x, foot_y = _court_to_px(H_inv, cx, cy)
            # Perspective box scaling: near players look taller than far players
            scale = 0.65 + 0.55 * (cy / 13.4)
            bw, bh = 74.0 * scale, 185.0 * scale
            bbox = [
                round(foot_x - bw / 2, 1),
                round(foot_y - bh, 1),
                round(foot_x + bw / 2, 1),
                round(foot_y, 1),
            ]
            frame_players[pid] = {
                "bbox_px": bbox,
                "foot_px": [foot_x, foot_y],
                "foot_court_m": [cx, cy],
                "speed_mps": spd,
                "dist_cum_m": round(dist_cum[pid], 2),
                "energy": energy_cur[pid],
                "conf": 0.94,
            }

        frames.append({"i": i, "t": t, "shuttle": shuttle, "players": frame_players})

    summary: Dict[str, Any] = {}
    for p in players_meta:
        pid = p["id"]
        zc = zone_counts[pid]
        summary[pid] = {
            "total_distance_m": round(dist_cum[pid], 2),
            "avg_speed_mps": round(sum(speeds_log[pid]) / len(speeds_log[pid]), 2),
            "max_speed_mps": round(max(speeds_log[pid]), 2),
            "zones_pct": {
                "front": round(100.0 * zc["front"] / total_frames, 1),
                "mid": round(100.0 * zc["mid"] / total_frames, 1),
                "back": round(100.0 * zc["back"] / total_frames, 1),
            },
        }

    return {
        "schema_version": "2.0",
        "match_id": "mock_match_001",
        "video": {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "width": width,
            "height": height,
        },
        "court": {
            "mode": "doubles",
            "dims_m": {"width": 6.1, "length": 13.4, "net_y": 6.7},
            "calib_points": calib,
            "homography_img_to_court": h_flat,
            "reproj_error_m": 0.04,
        },
        "players": players_meta,
        "rallies": rallies,
        "shots": shots,
        "frames": frames,
        "summary": summary,
    }


def generate_mock_video(data: Dict[str, Any], out_path: pathlib.Path) -> None:
    fps = int(data["video"]["fps"])
    # Render at half resolution (443x960) for fast encoding while keeping exact 886:1920 aspect
    vw, vh = 443, 960
    sx, sy = vw / data["video"]["width"], vh / data["video"]["height"]
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (vw, vh))

    calib, _, H_inv = _build_homography()
    # Pre-render static court background
    bg = np.full((vh, vw, 3), (38, 46, 36), dtype=np.uint8)
    court_poly = np.array(
        [
            _court_to_px(H_inv, 0.0, 0.0),
            _court_to_px(H_inv, 6.1, 0.0),
            _court_to_px(H_inv, 6.1, 13.4),
            _court_to_px(H_inv, 0.0, 13.4),
        ],
        dtype=np.float64,
    )
    court_poly[:, 0] *= sx
    court_poly[:, 1] *= sy
    cv2.fillPoly(bg, [court_poly.astype(np.int32)], (75, 125, 48))

    # Draw court boundary, net, and service lines
    lines_m = [
        ((0.0, 0.0), (6.1, 0.0)),
        ((6.1, 0.0), (6.1, 13.4)),
        ((6.1, 13.4), (0.0, 13.4)),
        ((0.0, 13.4), (0.0, 0.0)),
        ((0.46, 0.0), (0.46, 13.4)),
        ((5.64, 0.0), (5.64, 13.4)),
        ((0.0, 6.7), (6.1, 6.7)),
        ((0.0, 4.72), (6.1, 4.72)),
        ((0.0, 8.68), (6.1, 8.68)),
        ((3.05, 0.0), (3.05, 4.72)),
        ((3.05, 8.68), (3.05, 13.4)),
    ]
    for (x1, y1), (x2, y2) in lines_m:
        p1 = _court_to_px(H_inv, x1, y1)
        p2 = _court_to_px(H_inv, x2, y2)
        pt1 = (int(p1[0] * sx), int(p1[1] * sy))
        pt2 = (int(p2[0] * sx), int(p2[1] * sy))
        col = (210, 240, 255) if y1 == 6.7 and y2 == 6.7 else (235, 245, 235)
        cv2.line(bg, pt1, pt2, col, 2)

    for frame in data["frames"]:
        img = bg.copy()
        for pid, pf in frame["players"].items():
            bx1, by1, bx2, by2 = pf["bbox_px"]
            cx = int((bx1 + bx2) * 0.5 * sx)
            cy = int((by1 + by2) * 0.5 * sy)
            bw = max(8, int((bx2 - bx1) * 0.35 * sx))
            bh = max(14, int((by2 - by1) * 0.42 * sy))
            body_col = (210, 190, 90) if pid in ("p1", "p2") else (180, 140, 230)
            cv2.ellipse(img, (cx, cy), (bw, bh), 0, 0, 360, body_col, -1)
            cv2.circle(img, (cx, int(by1 * sy) + 8), 7, (235, 220, 200), -1)
        writer.write(img)
    writer.release()


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    out = root / "fixtures" / "mock_match.json"
    data = generate_mock_dict(seed=42)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    vid_out = root / "fixtures" / "mock_video.webm"
    generate_mock_video(data, vid_out)
    print(f"Generated {out} and {vid_out} ({len(data['frames'])} frames, {len(data['shots'])} shots)")


if __name__ == "__main__":
    main()
