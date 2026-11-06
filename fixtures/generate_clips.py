from __future__ import annotations
import json
import math
import pathlib
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def _make_homography(corners_px: List[List[float]], width_m: float = 6.1, length_m: float = 13.4) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    src = np.array(corners_px, dtype=np.float64)
    dst = np.array([[0.0, 0.0], [width_m, 0.0], [0.0, length_m], [width_m, length_m]], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    H_inv = np.linalg.inv(H)

    def to_px(xm: float, ym: float) -> List[float]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        return [round(float(v[0] / v[2]), 2), round(float(v[1] / v[2]), 2)]

    # 6 calibration points: 4 corners + 2 net posts (Y = 6.7m) + 2 service-line intersections
    calib = [
        {"px": to_px(0.0, 0.0), "court_m": [0.0, 0.0]},
        {"px": to_px(width_m, 0.0), "court_m": [width_m, 0.0]},
        {"px": to_px(0.0, 6.7), "court_m": [0.0, 6.7]},
        {"px": to_px(width_m, 6.7), "court_m": [width_m, 6.7]},
        {"px": to_px(0.0, 13.4), "court_m": [0.0, 13.4]},
        {"px": to_px(width_m, 13.4), "court_m": [width_m, 13.4]},
    ]
    return H, H_inv, calib


def _build_shuttle_and_shots(
    H_inv: np.ndarray,
    trajs: Dict[str, List[Tuple[float, float]]],
    n_frames: int,
    fps: int = 30,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def c2p(xm: float, ym: float) -> Tuple[float, float]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        return round(float(v[0] / v[2]), 2), round(float(v[1] / v[2]), 2)

    pids = list(trajs.keys())
    near_pids = [p for p in pids if p in ("p1", "p2")]
    far_pids = [p for p in pids if p in ("p3", "p4")]
    # Shot keyframes every 12 frames: 6, 18, 30, 42, 54
    shot_frames = [6, 18, 30, 42, 54]
    order = [
        near_pids[0],
        far_pids[0],
        near_pids[-1],
        far_pids[-1],
        near_pids[0],
    ]
    speeds_kmh = [168.0, 92.0, 64.0, 118.0, 84.0]
    gt_shots: List[Dict[str, Any]] = []
    for idx, sf in enumerate(shot_frames):
        pid = order[idx]
        next_pid = order[(idx + 1) % len(order)]
        next_f = shot_frames[idx + 1] if idx + 1 < len(shot_frames) else n_frames - 2
        fx, fy = trajs[pid][sf]
        tx, ty = trajs[next_pid][next_f]
        gt_shots.append(
            {
                "idx": idx + 1,
                "rally_id": 1,
                "frame": sf,
                "player_id": pid,
                "gt_speed_kmh": speeds_kmh[idx],
                "from_court_m": [round(fx, 2), round(fy, 2)],
                "to_court_m": [round(tx, 2), round(ty, 2)],
            }
        )

    gt_shuttle: List[Dict[str, Any]] = []
    for i in range(n_frames):
        if i < shot_frames[0] or i > n_frames - 2:
            gt_shuttle.append({"frame": i, "x_px": None, "y_px": None, "visible": False})
            continue
        active = gt_shots[0]
        next_f = n_frames - 2
        for s_idx, s in enumerate(gt_shots):
            if i >= s["frame"]:
                active = s
                next_f = gt_shots[s_idx + 1]["frame"] if s_idx + 1 < len(gt_shots) else n_frames - 2
        span = max(1, next_f - active["frame"])
        alpha = max(0.0, min(1.0, (i - active["frame"]) / span))
        xm = active["from_court_m"][0] * (1.0 - alpha) + active["to_court_m"][0] * alpha
        ym = active["from_court_m"][1] * (1.0 - alpha) + active["to_court_m"][1] * alpha
        px, py = c2p(xm, ym)
        arc = 36.0 * 4.0 * alpha * (1.0 - alpha)
        gt_shuttle.append({"frame": i, "x_px": round(px, 2), "y_px": round(py - arc, 2), "visible": True})

    return gt_shuttle, gt_shots


def _render_clip(
    mp4_path: pathlib.Path,
    H_inv: np.ndarray,
    trajectories_m: Dict[str, List[Tuple[float, float]]],
    gt_shuttle: Optional[List[Dict[str, Any]]] = None,
    width_m: float = 6.1,
    fps: int = 30,
    extra_bg_person: bool = False,
) -> None:
    vw, vh = 640, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (vw, vh))

    def c2p(xm: float, ym: float) -> Tuple[int, int]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    bg = np.full((vh, vw, 3), (35, 42, 35), dtype=np.uint8)
    poly = np.array([c2p(0.0, 0.0), c2p(width_m, 0.0), c2p(width_m, 13.4), c2p(0.0, 13.4)], dtype=np.int32)
    cv2.fillPoly(bg, [poly], (68, 125, 52))
    for (x1, y1), (x2, y2) in [
        ((0.0, 0.0), (width_m, 0.0)),
        ((width_m, 0.0), (width_m, 13.4)),
        ((width_m, 13.4), (0.0, 13.4)),
        ((0.0, 13.4), (0.0, 0.0)),
        ((0.0, 6.7), (width_m, 6.7)),
    ]:
        cv2.line(bg, c2p(x1, y1), c2p(x2, y2), (240, 245, 240), 2)

    total_frames = len(next(iter(trajectories_m.values())))
    colors = {
        "p1": (238, 211, 34),
        "p2": (21, 204, 250),
        "p3": (250, 139, 167),
        "p4": (128, 222, 74),
    }

    for f_idx in range(total_frames):
        img = bg.copy()
        # Optional bystander outside court polygon to test polygon filtering
        if extra_bg_person:
            cv2.rectangle(img, (20, 220), (52, 310), (120, 120, 200), -1)
        for pid, pts in trajectories_m.items():
            xm, ym = pts[f_idx]
            fx, fy = c2p(xm, ym)
            scale = 0.65 + 0.45 * (ym / 13.4)
            bw, bh = int(28 * scale), int(72 * scale)
            col = colors.get(pid, (200, 200, 200))
            # Draw high-contrast player silhouette with bright ankle markers
            cv2.rectangle(img, (fx - bw // 2, fy - bh), (fx + bw // 2, fy - 4), col, -1)
            cv2.circle(img, (fx, fy - bh - 8), max(5, int(7 * scale)), (240, 220, 200), -1)
            cv2.circle(img, (fx - 5, fy - 2), 3, (255, 255, 255), -1)
            cv2.circle(img, (fx + 5, fy - 2), 3, (255, 255, 255), -1)
        if gt_shuttle and f_idx < len(gt_shuttle) and gt_shuttle[f_idx]["visible"]:
            sx = int(round(gt_shuttle[f_idx]["x_px"]))
            sy = int(round(gt_shuttle[f_idx]["y_px"]))
            # Draw bright yellow-white shuttlecock blob (radius 5px)
            cv2.circle(img, (sx, sy), 5, (0, 255, 255), -1)
            cv2.circle(img, (sx, sy), 2, (255, 255, 255), -1)
        writer.write(img)
    writer.release()


def generate_all_fixtures() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    clips_dir = root / "fixtures" / "clips"
    labels_dir = root / "fixtures" / "labels"
    clips_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("synthetic_dev_01", "singles", 5.18, [[180, 160], [460, 160], [90, 560], [550, 560]], 60, False),
        ("synthetic_dev_02", "doubles", 6.1, [[165, 175], [475, 175], [75, 575], [565, 575]], 60, True),
        ("synthetic_dev_03", "doubles", 6.1, [[195, 240], [445, 240], [60, 590], [580, 590]], 60, True),
        ("synthetic_test_01", "singles", 5.18, [[175, 165], [465, 165], [85, 565], [555, 565]], 60, False),
        ("synthetic_test_02", "doubles", 6.1, [[170, 180], [470, 180], [70, 580], [570, 580]], 60, True),
    ]

    for name, mode, width_m, corners_px, n_frames, bg_person in specs:
        mp4_path = clips_dir / f"{name}.mp4"
        lbl_path = labels_dir / f"{name}.json"
        if "test_" in name and mp4_path.exists() and lbl_path.exists():
            continue

        _, H_inv, calib = _make_homography(corners_px, width_m=width_m)
        if mode == "singles":
            anchors = {"p1": (2.6, 10.2), "p3": (2.6, 3.2)}
            players_meta = [
                {"id": "p1", "name": "Player A", "team": "A", "color": "#22d3ee"},
                {"id": "p3", "name": "Player B", "team": "B", "color": "#a78bfa"},
            ]
        else:
            anchors = {"p1": (1.6, 10.3), "p2": (4.5, 10.6), "p3": (1.6, 3.1), "p4": (4.5, 2.8)}
            players_meta = [
                {"id": "p1", "name": "Foam", "team": "A", "color": "#22d3ee"},
                {"id": "p2", "name": "Kotty", "team": "A", "color": "#facc15"},
                {"id": "p3", "name": "Opp 1", "team": "B", "color": "#a78bfa"},
                {"id": "p4", "name": "Opp 2", "team": "B", "color": "#4ade80"},
            ]

        trajs: Dict[str, List[Tuple[float, float]]] = {pid: [] for pid in anchors}
        for i in range(n_frames):
            t = i / 30.0
            for idx_p, (pid, (ax, ay)) in enumerate(anchors.items()):
                xm = round(ax + 0.45 * math.sin(2 * math.pi * 0.4 * t + idx_p), 3)
                ym = round(ay + 0.55 * math.cos(2 * math.pi * 0.35 * t + idx_p), 3)
                trajs[pid].append((xm, ym))

        gt_shuttle, gt_shots = _build_shuttle_and_shots(H_inv, trajs, n_frames)
        _render_clip(mp4_path, H_inv, trajs, gt_shuttle=gt_shuttle, width_m=width_m, extra_bg_person=bg_person)

        initial_assignments = []
        for p in players_meta:
            xm, ym = trajs[p["id"]][0]
            v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
            initial_assignments.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "team": p["team"],
                    "color": p["color"],
                    "click_px": [round(float(v[0] / v[2]), 1), round(float(v[1] / v[2]), 1)],
                }
            )

        label_doc = {
            "clip": f"fixtures/clips/{name}.mp4",
            "mode": mode,
            "dims_m": {"width": width_m, "length": 13.4, "net_y": 6.7},
            "calib_points": calib,
            "initial_players": initial_assignments,
            "gt_trajectories_m": {pid: [list(pt) for pt in pts] for pid, pts in trajs.items()},
            "gt_shuttle_px": gt_shuttle,
            "gt_shots": gt_shots,
        }
        lbl_path.write_text(json.dumps(label_doc, indent=2) + "\n", encoding="utf-8")

    # Generate synthetic_walk_test.mp4: 1 player walking straight along sideline X=0.0, Y=0.0 -> 13.4m in 90 frames (3.0s)
    _, H_inv_w, calib_w = _make_homography([[165, 175], [475, 175], [75, 575], [565, 575]], width_m=6.1)
    walk_traj = {"p1": [(0.0, round(13.4 * i / 89.0, 4)) for i in range(90)]}
    walk_mp4 = root / "fixtures" / "synthetic_walk_test.mp4"
    _render_clip(walk_mp4, H_inv_w, walk_traj, width_m=6.1, extra_bg_person=False)
    walk_label = {
        "clip": "fixtures/synthetic_walk_test.mp4",
        "mode": "doubles",
        "dims_m": {"width": 6.1, "length": 13.4, "net_y": 6.7},
        "calib_points": calib_w,
        "expected_distance_m": 13.4,
        "gt_trajectories_m": {"p1": [list(pt) for pt in walk_traj["p1"]]},
    }
    (labels_dir / "synthetic_walk_test.json").write_text(json.dumps(walk_label, indent=2) + "\n", encoding="utf-8")
    print("Generated synthetic_dev_01..03, synthetic_test_01..02, synthetic_walk_test.mp4, and labels.")


if __name__ == "__main__":
    generate_all_fixtures()

