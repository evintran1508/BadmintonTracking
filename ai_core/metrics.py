from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter
import yaml
from ai_core.players import track_players_in_video


def load_metrics_config() -> Dict[str, Any]:
    root = pathlib.Path(__file__).resolve().parent.parent
    cfg_path = root / "config" / "metrics.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _smooth_trajectory(
    pts: List[Optional[Tuple[float, float]]],
    fps: float,
    max_speed_mps: float = 10.0,
) -> List[Optional[Tuple[float, float]]]:
    n = len(pts)
    if n == 0:
        return []
    xs = np.array([p[0] if p is not None else np.nan for p in pts], dtype=np.float64)
    ys = np.array([p[1] if p is not None else np.nan for p in pts], dtype=np.float64)
    idx = np.arange(n)
    valid_any = np.any(~np.isnan(xs))
    if not valid_any:
        return [None] * n

    for arr in (xs, ys):
        valid = ~np.isnan(arr)
        arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])

    # Clamp physical impossible single-frame jump noise (max sprinting speed)
    max_step = max_speed_mps / fps
    for i in range(1, n):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        step = math.hypot(dx, dy)
        if step > max_step:
            scale = max_step / step
            xs[i] = xs[i - 1] + dx * scale
            ys[i] = ys[i - 1] + dy * scale

    win = min(7, n if n % 2 == 1 else n - 1)
    if win >= 5:
        xs = savgol_filter(xs, window_length=win, polyorder=2, mode="interp")
        ys = savgol_filter(ys, window_length=win, polyorder=2, mode="interp")

    return [(round(float(xs[i]), 4), round(float(ys[i]), 4)) for i in range(n)]


def compute_player_metrics(
    raw_trajectories_m: Dict[str, List[Optional[Tuple[float, float]]]],
    fps: float = 30.0,
    shot_frames_by_player: Optional[Dict[str, List[int]]] = None,
) -> Tuple[List[Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    cfg = load_metrics_config()
    net_y = float(cfg["court"]["net_y_m"])
    front_max = float(cfg["court"]["zones_from_net_m"]["front_max"])
    mid_max = float(cfg["court"]["zones_from_net_m"]["mid_max"])
    e_init = float(cfg["energy"]["initial"])
    e_min = float(cfg["energy"]["min"])
    dist_decay = float(cfg["energy"]["dist_decay_per_m"])
    shot_decay = float(cfg["energy"]["shot_decay"])

    total_frames = len(next(iter(raw_trajectories_m.values()))) if raw_trajectories_m else 0
    frames_out: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(total_frames)]
    summary: Dict[str, Dict[str, Any]] = {}

    for pid, raw_pts in raw_trajectories_m.items():
        smoothed = _smooth_trajectory(raw_pts, fps=fps, max_speed_mps=10.0)
        dist_cum = 0.0
        energy = e_init
        speeds: List[float] = []
        zones = {"front": 0, "mid": 0, "back": 0}
        p_shots = set((shot_frames_by_player or {}).get(pid, []))

        for i, pt in enumerate(smoothed):
            if pt is None:
                frames_out[i][pid] = {
                    "foot_court_m": None,
                    "speed_mps": 0.0,
                    "dist_cum_m": round(dist_cum, 2),
                    "energy": energy,
                }
                continue
            xm, ym = pt
            if i > 0 and smoothed[i - 1] is not None:
                px_m, py_m = smoothed[i - 1]
                step_m = math.hypot(xm - px_m, ym - py_m)
            else:
                step_m = 0.0
            spd = round(step_m * fps, 2)
            dist_cum = round(dist_cum + step_m, 4)
            s_pen = shot_decay if i in p_shots else 0.0
            energy = round(max(e_min, energy - step_m * dist_decay - s_pen), 2)
            speeds.append(spd)

            d_net = abs(ym - net_y)
            if d_net <= front_max:
                zones["front"] += 1
            elif d_net <= mid_max:
                zones["mid"] += 1
            else:
                zones["back"] += 1

            frames_out[i][pid] = {
                "foot_court_m": [round(xm, 3), round(ym, 3)],
                "speed_mps": spd,
                "dist_cum_m": round(dist_cum, 2),
                "energy": energy,
            }

        denom = max(1, total_frames)
        summary[pid] = {
            "total_distance_m": round(dist_cum, 2),
            "avg_speed_mps": round(sum(speeds) / denom, 2),
            "max_speed_mps": round(max(speeds) if speeds else 0.0, 2),
            "zones_pct": {
                "front": round(100.0 * zones["front"] / denom, 1),
                "mid": round(100.0 * zones["mid"] / denom, 1),
                "back": round(100.0 * zones["back"] / denom, 1),
            },
        }

    return frames_out, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute smoothed player trajectories and metrics.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    calib_data = json.loads(pathlib.Path(args.calib).read_text(encoding="utf-8"))
    tracked = track_players_in_video(pathlib.Path(args.video), calib_data)

    raw_trajs: Dict[str, List[Optional[Tuple[float, float]]]] = {}
    if calib_data.get("initial_players"):
        for p in calib_data["initial_players"]:
            pid = p["id"]
            raw_trajs[pid] = [
                tuple(f.get("players", {}).get(pid, {}).get("foot_court_m"))
                if pid in f.get("players", {})
                else None
                for f in tracked["frames"]
            ]
    else:
        raw_trajs["p1"] = [
            tuple(f["detections"][0]["foot_court_m"]) if f["detections"] else None
            for f in tracked["frames"]
        ]

    frames_metrics, summary = compute_player_metrics(raw_trajs, fps=30.0)
    out_frames = [{"i": idx, "players": f_map} for idx, f_map in enumerate(frames_metrics)]
    out_doc = {"video": args.video, "frames": out_frames, "summary": summary}

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics computed for {args.video}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

