from __future__ import annotations
import json
import pathlib
import shutil
import sys
from typing import Any, Dict, List
import cv2
import numpy as np
from ai_core.court import calibrate_court
from schema.models import MatchDocument


def build_reference_assets(src_video: pathlib.Path, root: pathlib.Path) -> None:
    ref_dir = root / "fixtures" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    dst_video = ref_dir / "reference_match.mp4"
    if src_video.resolve() != dst_video.resolve():
        shutil.copy2(src_video, dst_video)

    cap = cv2.VideoCapture(str(dst_video))
    if not cap.isOpened():
        raise ValueError(f"Không mở được video mẫu: {dst_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 369)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 480)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1040)

    all_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()
    total_frames = len(all_frames)

    # Extract 12 reference images named by second: sec_00.jpg .. sec_11.jpg
    for sec in range(12):
        f_idx = min(total_frames - 1, int(round(sec * fps)))
        img_out = ref_dir / f"sec_{sec:02d}.jpg"
        cv2.imwrite(str(img_out), all_frames[f_idx], [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # Court homography matching portrait broadcast perspective (480x1040)
    corners_px = np.array(
        [[112.0, 495.0], [368.0, 495.0], [28.0, 815.0], [452.0, 815.0]],
        dtype=np.float64,
    )
    corners_m = np.array([[0.0, 0.0], [6.1, 0.0], [0.0, 13.4], [6.1, 13.4]], dtype=np.float64)
    H, _ = cv2.findHomography(corners_px, corners_m)
    H_inv = np.linalg.inv(H)

    def c2p(xm: float, ym: float) -> List[float]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        return [round(float(v[0] / v[2]), 1), round(float(v[1] / v[2]), 1)]

    calib_points = [
        {"px": c2p(0.0, 0.0), "court_m": [0.0, 0.0]},
        {"px": c2p(6.1, 0.0), "court_m": [6.1, 0.0]},
        {"px": c2p(0.0, 6.7), "court_m": [0.0, 6.7]},
        {"px": c2p(6.1, 6.7), "court_m": [6.1, 6.7]},
        {"px": c2p(0.0, 13.4), "court_m": [0.0, 13.4]},
        {"px": c2p(6.1, 13.4), "court_m": [6.1, 13.4]},
    ]
    court_meta = calibrate_court(calib_points, mode="doubles")

    players_meta = [
        {"id": "p1", "name": "Foam", "team": "A", "color": "#22d3ee"},
        {"id": "p2", "name": "Blue", "team": "A", "color": "#4ade80"},
        {"id": "p3", "name": "Opp 1", "team": "B", "color": "#c084fc"},
        {"id": "p4", "name": "Kotty", "team": "B", "color": "#facc15"},
    ]

    # Exact 13 shots reproducing Plan v3 Section 4 table across seconds 0..11:
    # Sec 0->2(214), 1->4(86), 2->5(24), 3->6(91), 4->7(57), 5->8(49),
    # 6->9(30), 7->9(30), 8->11(86), 9->11(86), 10->12(59), 11->13(51)
    shot_specs = [
        (0, "p3", None, [2.0, 3.2], [1.8, 10.2]),
        (0, "p1", 214.0, [1.8, 10.2], [4.3, 3.0]),
        (15, "p4", None, [4.3, 3.0], [4.2, 9.8]),
        (28, "p2", 86.0, [4.2, 9.8], [1.9, 3.5]),
        (55, "p3", 24.0, [1.9, 3.5], [2.2, 8.2]),
        (85, "p1", 91.0, [2.2, 8.2], [4.4, 2.9]),
        (115, "p4", 57.0, [4.4, 2.9], [1.7, 10.4]),
        (145, "p1", 49.0, [1.7, 10.4], [2.1, 3.6]),
        (175, "p3", 30.0, [2.1, 3.6], [4.1, 10.8]),
        (220, "p2", None, [4.1, 10.8], [4.0, 3.3]),
        (236, "p4", 86.0, [4.0, 3.3], [1.9, 10.5]),
        (292, "p1", 59.0, [1.9, 10.5], [2.3, 3.4]),
        (322, "p3", 51.0, [2.3, 3.4], [4.2, 10.1]),
    ]
    shots = [
        {
            "idx": i + 1,
            "rally_id": 1,
            "frame": f_num,
            "player_id": pid,
            "speed_kmh": spd,
            "speed_confidence": 0.9 if spd is not None else 0.0,
            "from_court_m": frm,
            "to_court_m": to,
        }
        for i, (f_num, pid, spd, frm, to) in enumerate(shot_specs)
    ]

    # Rally started 2.77s before clip opened (00:02.77 at t=0s -> 00:13.77 at t=11s)
    rallies = [
        {
            "id": 1,
            "start_frame": 0,
            "end_frame": total_frames - 1,
            "shot_count": 13,
        }
    ]

    base_pos = {
        "p1": (1.8, 10.4),
        "p2": (4.3, 9.9),
        "p3": (1.9, 3.3),
        "p4": (4.3, 3.1),
    }
    frames_doc: List[Dict[str, Any]] = []
    max_t = max(11.0, (total_frames - 1) / fps)

    for i in range(total_frames):
        t_sec = i / fps
        prog_11s = min(1.0, t_sec / 11.0)
        # Near player p1: energy 94 -> 61, MOVE 3.8m -> 26.7m over 0..11s (strictly monotonic)
        p1_energy = round(94.0 - (94.0 - 61.0) * (t_sec / max_t), 2)
        p1_dist = round(3.8 + (26.7 - 3.8) * (t_sec / max_t), 2)

        p_out: Dict[str, Any] = {}
        for idx_p, p in enumerate(players_meta):
            pid = p["id"]
            bx, by = base_pos[pid]
            fx_px, fy_px = c2p(bx, by)
            bw, bh = (52.0, 140.0) if p["team"] == "A" else (38.0, 102.0)
            e_val = p1_energy if pid == "p1" else round(95.0 - (18.0 + idx_p * 2.0) * (t_sec / max_t), 2)
            d_val = p1_dist if pid == "p1" else round(2.5 + (16.0 + idx_p * 1.5) * (t_sec / max_t), 2)
            p_out[pid] = {
                "bbox_px": [round(fx_px - bw / 2, 1), round(fy_px - bh, 1), round(fx_px + bw / 2, 1), round(fy_px, 1)],
                "foot_px": [fx_px, fy_px],
                "foot_court_m": [bx, by],
                "speed_mps": 2.1,
                "dist_cum_m": d_val,
                "energy": e_val,
                "conf": 0.95,
            }

        frames_doc.append(
            {
                "i": i,
                "t": round(t_sec, 4),
                "shuttle": {
                    "x_px": round(vw * 0.5, 1),
                    "y_px": round(vh * (0.35 + 0.12 * prog_11s), 1),
                    "visible": True,
                    "conf": 0.9,
                },
                "players": p_out,
            }
        )

    summary = {
        p["id"]: {
            "total_distance_m": frames_doc[-1]["players"][p["id"]]["dist_cum_m"],
            "avg_speed_mps": 2.1,
            "max_speed_mps": 4.2,
            "zones_pct": {"front": 20.0, "mid": 50.0, "back": 30.0},
        }
        for p in players_meta
    }

    doc = {
        "schema_version": "2.0",
        "match_id": "reference_hud_v3",
        "video": {
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 3),
            "width": vw,
            "height": vh,
        },
        "court": {
            "mode": "doubles",
            "dims_m": court_meta["dims_m"],
            "calib_points": court_meta["calib_points"],
            "homography_img_to_court": court_meta["homography_img_to_court"],
            "reproj_error_m": court_meta["reproj_error_m"],
        },
        "players": players_meta,
        "rallies": rallies,
        "shots": shots,
        "frames": frames_doc,
        "summary": summary,
    }
    MatchDocument.model_validate(doc)
    out_json = root / "fixtures" / "reference_hud.json"
    out_json.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Extracted 12 frames -> {ref_dir} and validated {out_json}")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (root / "fixtures" / "reference" / "reference_match.mp4")
    build_reference_assets(src, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

