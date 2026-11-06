from __future__ import annotations
import argparse
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import cv2
import numpy as np
from ai_core.court import calibrate_court
from ai_core.metrics import compute_player_metrics
from ai_core.players import (
    assign_and_stitch_tracks,
    auto_init_players_from_detections,
    detect_players_yolo_frame,
    is_inside_court_polygon,
    px_to_court_m,
    require_yolo_pose_model,
    select_active_court_players,
    select_best_seed_frame_dets,
    select_foot_point,
    select_wrist_point,
)
from ai_core.shots import detect_shots_and_rallies
from ai_core.shuttle import require_tracknet_weights, track_shuttle_in_video
from schema.models import MatchDocument


def run_pipeline(
    video_path: pathlib.Path,
    calib_data: Dict[str, Any],
    match_id: str = "match_001",
    player_stride: int = 1,
    max_frames: int = 900,
    progress_cb: Optional[Callable[[int], None]] = None,
    player_detector_fn: Optional[Callable[[np.ndarray], List[Dict[str, Any]]]] = None,
    shuttle_detector_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    weights_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    # Fail immediately if real weights are missing (Rule 9: no silent fallback)
    yolo_model = None if player_detector_fn is not None else require_yolo_pose_model(weights_dir)
    if shuttle_detector_fn is None:
        require_tracknet_weights(weights_dir)

    def report_progress(pct: int, msg: Optional[str] = None) -> None:
        if progress_cb is not None:
            try:
                progress_cb(pct)
            except Exception:
                pass
        if msg:
            print(f"STATUS:{msg}", flush=True)

    report_progress(10, "Đang nạp mô hình & hiệu chỉnh tọa độ sân...")
    court_meta = calibrate_court(calib_data["calib_points"], mode=calib_data.get("mode", "doubles"))
    H = np.array(court_meta["homography_img_to_court"], dtype=np.float64).reshape(3, 3)
    H_inv = np.linalg.inv(H)
    dims_m = court_meta["dims_m"]
    expected_count = 4 if court_meta["mode"] == "doubles" else 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    est_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if vw <= 0 or vh <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions")

    eff_stride = max(1, player_stride)
    limit_frames = max_frames if (max_frames is not None and max_frames > 0) else float("inf")
    target_total = int(limit_frames) if limit_frames != float("inf") else (est_total if est_total > 0 else 300)
    target_total = max(1, target_total)
    raw_frames: List[List[Dict[str, Any]]] = []
    last_filtered: List[Dict[str, Any]] = []
    idx = 0

    while idx < limit_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % eff_stride == 0:
            dets = (
                player_detector_fn(frame)
                if player_detector_fn is not None
                else detect_players_yolo_frame(frame, yolo_model)
            )
            filtered: List[Dict[str, Any]] = []
            for d in dets:
                fx, fy, used_ankle = select_foot_point(d["bbox_px"], d.get("keypoints"))
                wx, wy = select_wrist_point(d["bbox_px"], d.get("keypoints"))
                cm = px_to_court_m(H, fx, fy)
                if not is_inside_court_polygon(cm, dims_m):
                    continue
                filtered.append(
                    {
                        "raw_track_id": d["raw_track_id"],
                        "bbox_px": [round(v, 1) for v in d["bbox_px"]],
                        "foot_px": [fx, fy],
                        "wrist_px": [wx, wy],
                        "foot_court_m": [cm[0], cm[1]],
                        "used_ankle": used_ankle,
                        "conf": d["conf"],
                    }
                )
            if len(filtered) > expected_count:
                filtered = select_active_court_players(filtered, dims_m, mode=court_meta["mode"])
            last_filtered = filtered

        cur_dets = [
            dict(item, conf=0.75 if idx % eff_stride != 0 else item["conf"])
            for item in last_filtered
        ]
        raw_frames.append(cur_dets)
        idx += 1
        if idx % 15 == 0 or idx == target_total or idx == limit_frames:
            pct = min(55, 10 + int(45 * idx / target_total))
            report_progress(pct, f"Bám người chơi [YOLO11n-Pose]: Khung hình {idx}/{target_total} ({pct}%)")
    cap.release()

    total_frames = idx
    if total_frames == 0:
        raise ValueError("Video contains 0 readable frames")

    initial_players = calib_data.get("initial_players") or []
    if not initial_players:
        seed_dets = select_best_seed_frame_dets(raw_frames, mode=court_meta["mode"], dims_m=dims_m)
        initial_players = auto_init_players_from_detections(seed_dets, mode=court_meta["mode"], dims_m=dims_m)

    report_progress(56, f"Khớp nối định danh người chơi ({total_frames} khung hình)...")
    assigned_per_frame, _switches = assign_and_stitch_tracks(raw_frames, initial_players, H)

    def shuttle_prog(done: int, tot: int) -> None:
        pct = min(94, 56 + int(38 * done / max(1, tot)))
        report_progress(pct, f"Dò quỹ đạo cầu [TrackNetV3]: Khung hình {done}/{tot} ({pct}%)")

    shuttle_res = track_shuttle_in_video(
        video_path,
        calib_data,
        detector_fn=shuttle_detector_fn,
        weights_dir=weights_dir,
        max_frames=total_frames,
        progress_cb=shuttle_prog if progress_cb is not None else None,
    )
    shuttle_frames = shuttle_res["frames"][:total_frames]
    player_frames_wrapped = [{"i": i, "players": assigned_per_frame[i]} for i in range(total_frames)]
    report_progress(95, "Phân tích cú đánh & nhịp rally...")
    shots, rallies = detect_shots_and_rallies(shuttle_frames, player_frames_wrapped, H, fps=fps)

    shot_frames_by_player: Dict[str, List[int]] = {}
    for s in shots:
        shot_frames_by_player.setdefault(s["player_id"], []).append(s["frame"])

    raw_trajs: Dict[str, List[Optional[Tuple[float, float]]]] = {}
    for p in initial_players:
        pid = p["id"]
        raw_trajs[pid] = [
            tuple(assigned_per_frame[i][pid]["foot_court_m"]) if pid in assigned_per_frame[i] else None
            for i in range(total_frames)
        ]

    frames_metrics, summary = compute_player_metrics(raw_trajs, fps=fps, shot_frames_by_player=shot_frames_by_player)

    def c2p(xm: float, ym: float) -> Tuple[float, float]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        w = float(v[2]) if abs(float(v[2])) > 1e-9 else 1e-9
        return round(float(v[0] / w), 1), round(float(v[1] / w), 1)

    frames_doc: List[Dict[str, Any]] = []
    for i in range(total_frames):
        sf = shuttle_frames[i] if i < len(shuttle_frames) else {"x_px": None, "y_px": None, "visible": False, "conf": 0.0}
        p_out: Dict[str, Any] = {}
        for p in initial_players:
            pid = p["id"]
            pm = frames_metrics[i].get(pid, {})
            raw_det = assigned_per_frame[i].get(pid)
            if raw_det and raw_det.get("foot_court_m") is not None:
                xm, ym = raw_det["foot_court_m"]
                fx, fy = raw_det["foot_px"]
                bbox = raw_det["bbox_px"]
                conf = raw_det["conf"]
            elif pm.get("foot_court_m") is not None:
                xm, ym = pm["foot_court_m"]
                fx, fy = c2p(xm, ym)
                bbox = None
                conf = 0.5
            else:
                xm, ym = None, None
                fx, fy = None, None
                bbox = None
                conf = 0.0

            p_out[pid] = {
                "bbox_px": bbox,
                "foot_px": [fx, fy] if fx is not None and fy is not None else None,
                "foot_court_m": [xm, ym] if xm is not None and ym is not None else None,
                "speed_mps": pm.get("speed_mps", 0.0),
                "dist_cum_m": pm.get("dist_cum_m", 0.0),
                "energy": pm.get("energy", 100.0),
                "conf": conf,
            }
        frames_doc.append(
            {
                "i": i,
                "t": round(i / fps, 4),
                "shuttle": {
                    "x_px": sf["x_px"],
                    "y_px": sf["y_px"],
                    "visible": sf["visible"],
                    "conf": sf["conf"],
                },
                "players": p_out,
            }
        )

    doc_dict = {
        "schema_version": "2.0",
        "match_id": match_id,
        "video": {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 3),
            "width": vw,
            "height": vh,
        },
        "court": {
            "mode": court_meta["mode"],
            "dims_m": court_meta["dims_m"],
            "calib_points": court_meta["calib_points"],
            "homography_img_to_court": court_meta["homography_img_to_court"],
            "reproj_error_m": court_meta["reproj_error_m"],
        },
        "players": [
            {"id": p["id"], "name": p["name"], "team": p["team"], "color": p.get("color", "#22d3ee")}
            for p in initial_players
        ],
        "rallies": rallies,
        "shots": shots,
        "frames": frames_doc,
        "summary": summary,
    }
    MatchDocument.model_validate(doc_dict)
    report_progress(100, f"Hoàn tất phân tích AI ({total_frames} khung hình, {len(shots)} cú đánh)")
    return doc_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full AI badminton pipeline to produce match.json.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--player-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--weights-dir", default=None)
    args = parser.parse_args()

    try:
        calib_data = json.loads(pathlib.Path(args.calib).read_text(encoding="utf-8"))
        wdir = pathlib.Path(args.weights_dir) if args.weights_dir else None
        doc = run_pipeline(
            pathlib.Path(args.video),
            calib_data,
            player_stride=args.player_stride,
            max_frames=args.max_frames,
            progress_cb=lambda pct: print(f"PROGRESS:{pct}", flush=True),
            weights_dir=wdir,
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Pipeline complete -> {out_path} ({doc['video']['total_frames']} frames, {len(doc['shots'])} shots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
