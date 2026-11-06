from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ai_core.court import calibrate_court

POSE_WEIGHTS_FILENAME = "yolo11n-pose.pt"
_YOLO_POSE_MODEL_CACHE: Dict[str, Any] = {}


def _ensure_torch_numpy_bridge() -> None:
    """Bridge PyTorch 2.3 (compiled against NumPy 1.x) with NumPy 2.x without package downgrades."""
    warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")
    import torch

    if getattr(torch, "_badminton_np_patched", False):
        return
    _orig = torch.from_numpy
    dtype_map = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("int64"): torch.int64,
        np.dtype("int32"): torch.int32,
        np.dtype("int16"): torch.int16,
        np.dtype("bool"): torch.bool,
    }

    def _safe_from_numpy(arr: np.ndarray) -> Any:
        try:
            return _orig(arr)
        except RuntimeError:
            arr_c = np.ascontiguousarray(arr)
            tdtype = dtype_map.get(arr_c.dtype, torch.float32)
            if arr_c.size == 0:
                return torch.empty(arr_c.shape, dtype=tdtype)
            return torch.frombuffer(bytearray(arr_c.tobytes()), dtype=tdtype).reshape(arr_c.shape).clone()

    import ctypes

    rev_dtype_map = {v: k for k, v in dtype_map.items()}

    def _safe_to_numpy(tensor: Any) -> np.ndarray:
        t_c = tensor.detach().cpu().contiguous()
        np_dt = rev_dtype_map.get(t_c.dtype)
        if np_dt is None:
            t_c = t_c.to(torch.float32)
            np_dt = np.dtype("float32")
        nbytes = t_c.numel() * t_c.element_size()
        if nbytes == 0:
            return np.empty(tuple(t_c.shape), dtype=np_dt)
        buf = (ctypes.c_char * nbytes).from_address(t_c.data_ptr())
        return np.frombuffer(buf, dtype=np_dt).reshape(tuple(t_c.shape)).copy()

    torch.from_numpy = _safe_from_numpy
    torch.Tensor.numpy = _safe_to_numpy
    torch._badminton_np_patched = True


def require_yolo_pose_model(weights_dir: Optional[pathlib.Path] = None) -> Any:
    """Load real YOLO-Pose model; fail loudly if weights are missing (Rule 9: no silent fallback)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    wdir = weights_dir if weights_dir is not None else (root / "weights")
    weights_file = wdir / POSE_WEIGHTS_FILENAME
    if not weights_file.exists():
        raise FileNotFoundError(f"Thiếu weights/{POSE_WEIGHTS_FILENAME}, chạy make weights")
    key = str(weights_file.resolve())
    if key in _YOLO_POSE_MODEL_CACHE:
        return _YOLO_POSE_MODEL_CACHE[key]
    _ensure_torch_numpy_bridge()
    from ultralytics import YOLO

    import torch
    model = YOLO(key)
    if torch.cuda.is_available():
        try:
            model.to("cuda")
        except Exception:
            pass
    _YOLO_POSE_MODEL_CACHE[key] = model
    return model


def select_foot_point(
    bbox_px: List[float],
    keypoints: Optional[List[List[float]]] = None,
    min_kp_conf: float = 0.4,
) -> Tuple[float, float, bool]:
    """Select foot point from COCO ankle keypoints (15, 16) with fallback to bbox bottom-center."""
    if keypoints and len(keypoints) >= 17:
        valid_ankles = [
            (float(keypoints[idx][0]), float(keypoints[idx][1]))
            for idx in (15, 16)
            if len(keypoints[idx]) >= 3
            and float(keypoints[idx][2]) >= min_kp_conf
            and float(keypoints[idx][0]) > 1.0
            and float(keypoints[idx][1]) > 1.0
        ]
        if valid_ankles:
            fx = sum(p[0] for p in valid_ankles) / len(valid_ankles)
            fy = sum(p[1] for p in valid_ankles) / len(valid_ankles)
            return round(fx, 2), round(fy, 2), True

    x1, _y1, x2, y2 = bbox_px
    return round((x1 + x2) * 0.5, 2), round(float(y2), 2), False


def select_wrist_point(
    bbox_px: List[float],
    keypoints: Optional[List[List[float]]] = None,
    min_kp_conf: float = 0.35,
) -> Tuple[float, float]:
    """Select active racket/wrist keypoint (COCO 9, 10) or upper-body fallback."""
    if keypoints and len(keypoints) >= 11:
        wrists = [
            (float(keypoints[idx][0]), float(keypoints[idx][1]), float(keypoints[idx][2]))
            for idx in (9, 10)
            if len(keypoints[idx]) >= 3 and float(keypoints[idx][2]) >= min_kp_conf and float(keypoints[idx][0]) > 1.0
        ]
        if wrists:
            wrists.sort(key=lambda w: w[1])
            return round(wrists[0][0], 2), round(wrists[0][1], 2)
    x1, y1, x2, y2 = bbox_px
    return round((x1 + x2) * 0.5, 2), round(y1 + (y2 - y1) * 0.3, 2)


def px_to_court_m(H: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    vec = H @ np.array([px, py, 1.0], dtype=np.float64)
    w = float(vec[2]) if abs(float(vec[2])) > 1e-9 else 1e-9
    return round(float(vec[0] / w), 3), round(float(vec[1] / w), 3)


def is_inside_court_polygon(
    foot_court_m: Tuple[float, float],
    dims_m: Dict[str, float],
    margin_m: float = 0.85,
) -> bool:
    xm, ym = foot_court_m
    w = float(dims_m["width"])
    l = float(dims_m["length"])
    return (-margin_m <= xm <= w + margin_m) and (-margin_m <= ym <= l + margin_m)


def _court_interior_score(
    foot_court_m: List[float] | Tuple[float, float],
    dims_m: Optional[Dict[str, float]],
    conf: float = 0.8,
) -> float:
    xm, ym = float(foot_court_m[0]), float(foot_court_m[1])
    w = float(dims_m["width"]) if dims_m and "width" in dims_m else 6.1
    l = float(dims_m["length"]) if dims_m and "length" in dims_m else 13.4
    inside_strict = (-0.25 <= xm <= w + 0.25) and (-0.35 <= ym <= l + 0.35)
    dx = max(0.0, -xm, xm - w)
    dy = max(0.0, -ym, ym - l)
    return (10.0 if inside_strict else 0.0) + float(conf) - (dx + dy) * 2.5


def select_active_court_players(
    dets: List[Dict[str, Any]],
    dims_m: Dict[str, float],
    mode: str = "doubles",
) -> List[Dict[str, Any]]:
    """Prioritize players inside the court lines balanced across near/far half-courts over sideline spectators."""
    expected_count = 4 if mode == "doubles" else 2
    if len(dets) <= expected_count:
        return list(dets)
    per_side = 2 if mode == "doubles" else 1
    scored = sorted(
        dets,
        key=lambda d: _court_interior_score(d["foot_court_m"], dims_m, d.get("conf", 0.8)),
        reverse=True,
    )
    near = [d for d in scored if d["foot_court_m"][1] >= 6.7]
    far = [d for d in scored if d["foot_court_m"][1] < 6.7]
    chosen = near[:per_side] + far[:per_side]
    if len(chosen) < expected_count:
        chosen_ids = {id(d) for d in chosen}
        for d in scored:
            if id(d) not in chosen_ids:
                chosen.append(d)
                chosen_ids.add(id(d))
                if len(chosen) >= expected_count:
                    break
    return chosen[:expected_count]


def detect_players_yolo_frame(
    frame_bgr: np.ndarray,
    yolo_model: Any,
    imgsz: Optional[int] = None,
) -> List[Dict[str, Any]]:
    import torch
    vh, vw = frame_bgr.shape[:2]
    target_sz = imgsz if imgsz is not None else (1280 if max(vw, vh) >= 960 else 640)
    dev = 0 if torch.cuda.is_available() else "cpu"
    results = yolo_model.predict(frame_bgr, conf=0.18, iou=0.45, imgsz=target_sz, device=dev, verbose=False)
    dets: List[Dict[str, Any]] = []
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        kps = results[0].keypoints
        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()
        cls_list = boxes.cls.tolist()
        kp_all = kps.data.tolist() if (kps is not None and kps.data is not None) else None
        for i in range(len(xyxy_list)):
            if int(cls_list[i]) != 0:
                continue
            b = [float(x) for x in xyxy_list[i]]
            conf = float(conf_list[i])
            kp_list = kp_all[i] if kp_all is not None and i < len(kp_all) else None
            dets.append({"raw_track_id": i + 1, "bbox_px": b, "keypoints": kp_list, "conf": round(conf, 3)})
    return dets


def auto_init_players_from_detections(
    first_frame_dets: List[Dict[str, Any]],
    mode: str = "doubles",
    dims_m: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Initialize 2 or 4 players from court detections, prioritizing on-court players over bench spectators."""
    target_n = 4 if mode == "doubles" else 2
    palette = ["#22d3ee", "#4ade80", "#c084fc", "#facc15"]
    default_names = ["Foam", "Blue", "Opp 1", "Kotty"] if mode == "doubles" else ["Player A", "Player B"]

    near = [d for d in first_frame_dets if d["foot_court_m"][1] >= 6.7]
    far = [d for d in first_frame_dets if d["foot_court_m"][1] < 6.7]
    near.sort(key=lambda d: _court_interior_score(d["foot_court_m"], dims_m, d.get("conf", 0.8)), reverse=True)
    far.sort(key=lambda d: _court_interior_score(d["foot_court_m"], dims_m, d.get("conf", 0.8)), reverse=True)

    per_side = 2 if mode == "doubles" else 1
    chosen_near = sorted(near[:per_side], key=lambda d: d["foot_court_m"][0])
    chosen_far = sorted(far[:per_side], key=lambda d: d["foot_court_m"][0])

    while len(chosen_near) < per_side:
        idx = len(chosen_near)
        chosen_near.append({"foot_px": [200.0 + idx * 180.0, 500.0], "foot_court_m": [1.8 + idx * 2.5, 10.2]})
    while len(chosen_far) < per_side:
        idx = len(chosen_far)
        chosen_far.append({"foot_px": [200.0 + idx * 180.0, 260.0], "foot_court_m": [1.8 + idx * 2.5, 3.2]})

    init_list: List[Dict[str, Any]] = []
    combined = [(d, "A") for d in chosen_near] + [(d, "B") for d in chosen_far]
    for idx, (det, team) in enumerate(combined[:target_n]):
        init_list.append(
            {
                "id": f"p{idx + 1}" if mode == "doubles" else ("p1" if idx == 0 else "p3"),
                "name": default_names[idx],
                "team": team,
                "color": palette[idx % len(palette)],
                "click_px": list(det["foot_px"]),
                "init_court_m": list(det["foot_court_m"]),
            }
        )
    return init_list


def select_best_seed_frame_dets(
    raw_frames: List[List[Dict[str, Any]]],
    mode: str = "doubles",
    dims_m: Optional[Dict[str, float]] = None,
    max_scan: int = 20,
) -> List[Dict[str, Any]]:
    """Scan early frames to pick the frame with the best balanced near/far player coverage."""
    if not raw_frames:
        return []
    per_side = 2 if mode == "doubles" else 1
    best_dets = raw_frames[0]
    best_score = -1e9
    for dets in raw_frames[:max_scan]:
        if not dets:
            continue
        near = sorted(
            [d for d in dets if d["foot_court_m"][1] >= 6.7],
            key=lambda d: _court_interior_score(d["foot_court_m"], dims_m, d.get("conf", 0.8)),
            reverse=True,
        )[:per_side]
        far = sorted(
            [d for d in dets if d["foot_court_m"][1] < 6.7],
            key=lambda d: _court_interior_score(d["foot_court_m"], dims_m, d.get("conf", 0.8)),
            reverse=True,
        )[:per_side]
        chosen = near + far
        score = len(chosen) * 20.0 + sum(
            _court_interior_score(d["foot_court_m"], dims_m, d.get("conf", 0.8)) for d in chosen
        )
        if score > best_score:
            best_score = score
            best_dets = dets
    return best_dets


def assign_and_stitch_tracks(
    frames_dets: List[List[Dict[str, Any]]],
    initial_players: List[Dict[str, Any]],
    H: np.ndarray,
) -> Tuple[List[Dict[str, Dict[str, Any]]], int]:
    """Assign player IDs from initial player coordinates and stitch tracks continuously by court position."""
    first_dets = next((fd for fd in frames_dets if fd), [])
    last_court_pos: Dict[str, Tuple[float, float]] = {}
    player_team: Dict[str, str] = {}
    used_initial_dets: set[int] = set()

    for p in initial_players:
        pid = p["id"]
        player_team[pid] = p["team"]
        if "init_court_m" in p and p["init_court_m"] is not None:
            last_court_pos[pid] = (float(p["init_court_m"][0]), float(p["init_court_m"][1]))
        elif "click_px" in p and first_dets:
            cpx, cpy = p["click_px"]
            best_det = None
            best_idx = -1
            min_dist = float("inf")
            for d_idx, d in enumerate(first_dets):
                if d_idx in used_initial_dets:
                    continue
                if "bbox_px" in d:
                    x1, y1, x2, y2 = d["bbox_px"]
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    dist_px = math.hypot(cpx - cx, cpy - cy)
                    if x1 <= cpx <= x2 and y1 <= cpy <= y2:
                        dist_px *= 0.35
                elif "foot_px" in d:
                    fx, fy = d["foot_px"]
                    dist_px = math.hypot(cpx - fx, cpy - fy)
                else:
                    dist_px = float("inf")

                if dist_px < min_dist:
                    min_dist = dist_px
                    best_det = d
                    best_idx = d_idx
            if best_det is not None and min_dist < 320.0:
                used_initial_dets.add(best_idx)
                last_court_pos[pid] = tuple(best_det["foot_court_m"])
            else:
                last_court_pos[pid] = px_to_court_m(H, float(cpx), float(cpy))
        elif "click_px" in p:
            cpx, cpy = p["click_px"]
            last_court_pos[pid] = px_to_court_m(H, float(cpx), float(cpy))

    assigned_frames: List[Dict[str, Dict[str, Any]]] = []
    id_switches = 0

    for f_dets in frames_dets:
        frame_map: Dict[str, Dict[str, Any]] = {}
        used_dets = set()
        pairs: List[Tuple[float, str, int]] = []
        for pid, prev_m in last_court_pos.items():
            for d_idx, det in enumerate(f_dets):
                xm, ym = det["foot_court_m"]
                if player_team.get(pid) == "A" and ym < 4.8:
                    continue
                if player_team.get(pid) == "B" and ym > 8.6:
                    continue
                dist = math.hypot(xm - prev_m[0], ym - prev_m[1])
                if dist <= 4.2:
                    pairs.append((dist, pid, d_idx))
        pairs.sort(key=lambda x: x[0])

        for dist, pid, d_idx in pairs:
            if pid in frame_map or d_idx in used_dets:
                continue
            used_dets.add(d_idx)
            det = f_dets[d_idx]
            if dist > 1.25:
                id_switches += 1
            last_court_pos[pid] = tuple(det["foot_court_m"])
            frame_map[pid] = det
        assigned_frames.append(frame_map)

    return assigned_frames, id_switches


def track_players_in_video(
    video_path: pathlib.Path,
    calib_data: Dict[str, Any],
    debug_video_path: Optional[pathlib.Path] = None,
    detector_fn: Optional[Callable[[np.ndarray], List[Dict[str, Any]]]] = None,
    weights_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    court_meta = calibrate_court(calib_data["calib_points"], mode=calib_data.get("mode", "doubles"))
    H = np.array(court_meta["homography_img_to_court"], dtype=np.float64).reshape(3, 3)
    dims_m = court_meta["dims_m"]
    expected_count = 4 if court_meta["mode"] == "doubles" else 2

    yolo_model = None if detector_fn is not None else require_yolo_pose_model(weights_dir)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(debug_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))

    raw_frames: List[List[Dict[str, Any]]] = []
    frames_out: List[Dict[str, Any]] = []
    full_count_frames = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector_fn(frame) if detector_fn is not None else detect_players_yolo_frame(frame, yolo_model)

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

        if len(filtered) >= expected_count:
            full_count_frames += 1
        raw_frames.append(filtered)
        frames_out.append({"i": idx, "detections": filtered})

        if writer is not None:
            dbg = frame.copy()
            for fd in filtered:
                x1, y1, x2, y2 = [int(v) for v in fd["bbox_px"]]
                fx, fy = [int(v) for v in fd["foot_px"]]
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 180), 2)
                cv2.circle(dbg, (fx, fy), 5, (0, 80, 255), -1)
            writer.write(dbg)
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    initial_players = calib_data.get("initial_players", [])
    if not initial_players and raw_frames:
        seed_dets = select_best_seed_frame_dets(raw_frames, mode=court_meta["mode"], dims_m=dims_m)
        initial_players = auto_init_players_from_detections(seed_dets, mode=court_meta["mode"], dims_m=dims_m)

    assigned_per_frame, id_switches = (
        assign_and_stitch_tracks(raw_frames, initial_players, H) if initial_players else ([], 0)
    )
    for f_idx, pmap in enumerate(assigned_per_frame):
        frames_out[f_idx]["players"] = pmap

    ratio = round(full_count_frames / max(1, idx), 4)
    return {
        "video": str(video_path),
        "mode": court_meta["mode"],
        "total_frames": idx,
        "coverage_4_players_ratio": ratio,
        "id_switches": id_switches,
        "initial_players": initial_players,
        "frames": frames_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track players with YOLO-Pose + court polygon filter.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug-video", default=None)
    parser.add_argument("--weights-dir", default=None)
    args = parser.parse_args()

    try:
        calib_data = json.loads(pathlib.Path(args.calib).read_text(encoding="utf-8"))
        dbg_path = pathlib.Path(args.debug_video) if args.debug_video else None
        wdir = pathlib.Path(args.weights_dir) if args.weights_dir else None
        result = track_players_in_video(pathlib.Path(args.video), calib_data, dbg_path, weights_dir=wdir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Tracked {args.video}: coverage={result['coverage_4_players_ratio']*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
