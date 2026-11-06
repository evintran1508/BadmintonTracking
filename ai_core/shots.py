from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ai_core.court import calibrate_court
from ai_core.players import px_to_court_m, track_players_in_video
from ai_core.shuttle import track_shuttle_in_video


def _nearest_player_at_frame(
    f_num: int,
    sx_px: float,
    sy_px: float,
    s_court: Tuple[float, float],
    player_frames: List[Dict[str, Any]],
) -> Tuple[str, float, float, Tuple[float, float]]:
    """Return (best_pid, best_court_dist_m, best_px_dist, best_pos_m) for a shuttle position."""
    p_map = player_frames[f_num].get("players", {}) if f_num < len(player_frames) else {}
    best_pid = "p1"
    best_court_dist = 999.0
    best_px_dist = 9999.0
    best_pos_m = s_court
    best_combined = 9999.0
    for pid, pdata in p_map.items():
        if not pdata.get("foot_court_m"):
            continue
        pm = tuple(pdata["foot_court_m"])
        d_court = math.hypot(pm[0] - s_court[0], pm[1] - s_court[1])
        d_px = 9999.0
        if pdata.get("wrist_px"):
            wx, wy = pdata["wrist_px"]
            d_px = math.hypot(wx - sx_px, wy - sy_px)
        elif pdata.get("foot_px"):
            fx, fy = pdata["foot_px"]
            d_px = math.hypot(fx - sx_px, (fy - 55.0) - sy_px)
        combined = d_court + min(d_px / 120.0, 2.5)
        if combined < best_combined:
            best_combined = combined
            best_court_dist = d_court
            best_px_dist = d_px
            best_pid = pid
            best_pos_m = pm
    return best_pid, best_court_dist, best_px_dist, best_pos_m


def _classify_shot_type(
    is_first_in_rally: bool,
    speed_kmh: Optional[float],
    from_m: Tuple[float, float],
    to_m: Tuple[float, float],
    dy_next_px: float,
) -> str:
    if is_first_in_rally and (speed_kmh is None or speed_kmh < 105.0):
        return "Serve"
    dist_to_net = abs(to_m[1] - 6.7)
    if speed_kmh is not None and speed_kmh >= 105.0 and dy_next_px >= -2.0:
        return "Smash"
    if dist_to_net <= 2.25 and (speed_kmh is None or speed_kmh < 90.0):
        return "Drop"
    if dist_to_net >= 4.0 and dy_next_px < 0:
        return "Clear"
    return "Drive"


def detect_shots_and_rallies(
    shuttle_frames: List[Dict[str, Any]],
    player_frames: List[Dict[str, Any]],
    H: np.ndarray,
    fps: float = 30.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(shuttle_frames)
    vis_indices = [i for i, sf in enumerate(shuttle_frames) if sf.get("visible") and sf.get("x_px") is not None]
    if not vis_indices:
        return [], []

    # 1. Segment visible indices into rallies when gap exceeds ~1.6s
    rally_gap_frames = max(36, int(fps * 1.6))
    rally_segments: List[List[int]] = [[vis_indices[0]]]
    for v_idx in vis_indices[1:]:
        if v_idx - rally_segments[-1][-1] > rally_gap_frames:
            rally_segments.append([v_idx])
        else:
            rally_segments[-1].append(v_idx)

    xs = [shuttle_frames[i]["x_px"] for i in range(n)]
    ys = [shuttle_frames[i]["y_px"] for i in range(n)]
    min_shot_gap = 6 if n <= 120 else max(8, int(fps * 0.30))

    shots: List[Dict[str, Any]] = []
    rallies: List[Dict[str, Any]] = []
    score_a = 0
    score_b = 0

    for r_idx, seg_vis in enumerate(rally_segments):
        rally_id = r_idx + 1
        start_f = seg_vis[0]
        end_f = seg_vis[-1]
        shot_frames: List[int] = [start_f]

        for idx in range(max(2, start_f + 2), min(n - 2, end_f - 1)):
            if not (
                shuttle_frames[idx - 2]["visible"]
                and shuttle_frames[idx]["visible"]
                and shuttle_frames[idx + 2]["visible"]
            ):
                continue
            dy_prev = float(ys[idx] - ys[idx - 2])
            dy_next = float(ys[idx + 2] - ys[idx])
            dx_prev = float(xs[idx] - xs[idx - 2])
            dx_next = float(xs[idx + 2] - xs[idx])

            y_rev = dy_prev * dy_next < 0 and (abs(dy_prev) + abs(dy_next)) > 6.0
            x_rev = dx_prev * dx_next < 0 and (abs(dx_prev) + abs(dx_next)) > 14.0
            if not (y_rev or x_rev):
                continue
            if idx - shot_frames[-1] < min_shot_gap:
                continue

            # Reject mid-air parabola peak: shuttle reaches highest point in image (dy_prev < 0, dy_next > 0)
            # while continuing in the same horizontal direction far from any player
            if dy_prev < 0 and dy_next > 0 and dx_prev * dx_next >= 0 and n > 120:
                sx_px, sy_px = float(xs[idx]), float(ys[idx])
                s_court = px_to_court_m(H, sx_px, sy_px)
                _, d_court, d_px, _ = _nearest_player_at_frame(idx, sx_px, sy_px, s_court, player_frames)
                if d_court > 2.8 and d_px > 145.0:
                    continue

            shot_frames.append(idx)

        rally_shots: List[Dict[str, Any]] = []
        for s_in_rally, f_num in enumerate(shot_frames):
            sf = shuttle_frames[f_num]
            sx_px, sy_px = float(sf["x_px"]), float(sf["y_px"])
            s_court = px_to_court_m(H, sx_px, sy_px)
            best_pid, _, _, best_pos_m = _nearest_player_at_frame(f_num, sx_px, sy_px, s_court, player_frames)

            next_f = shot_frames[s_in_rally + 1] if s_in_rally + 1 < len(shot_frames) else end_f
            next_sf = shuttle_frames[next_f]
            to_court = px_to_court_m(H, float(next_sf["x_px"]), float(next_sf["y_px"]))

            # Rule 9: No artificial speed multiplier, no clamping; return None when uncertain
            frame_span = next_f - f_num
            dist_m = math.hypot(to_court[0] - best_pos_m[0], to_court[1] - best_pos_m[1])
            if frame_span >= 2 and dist_m >= 0.5:
                dt_sec = frame_span / fps
                raw_kmh = (dist_m / dt_sec) * 3.6
                if 10.0 <= raw_kmh <= 450.0:
                    speed_kmh: Optional[float] = round(raw_kmh, 1)
                    conf = 0.70
                else:
                    speed_kmh = None
                    conf = 0.0
            else:
                speed_kmh = None
                conf = 0.0

            dy_next_px = float(ys[min(n - 1, f_num + 2)] - sy_px) if shuttle_frames[min(n - 1, f_num + 2)].get("visible") else 0.0
            shot_type = _classify_shot_type(s_in_rally == 0, speed_kmh, best_pos_m, to_court, dy_next_px)

            shot_obj = {
                "idx": len(shots) + 1,
                "rally_id": rally_id,
                "frame": f_num,
                "player_id": best_pid,
                "shot_type": shot_type,
                "speed_kmh": speed_kmh,
                "speed_confidence": conf,
                "from_court_m": [round(best_pos_m[0], 2), round(best_pos_m[1], 2)],
                "to_court_m": [round(to_court[0], 2), round(to_court[1], 2)],
            }
            shots.append(shot_obj)
            rally_shots.append(shot_obj)

        # Infer rally winner_team ("A" or "B") from last shot & terminal shuttle landing
        last_shot = rally_shots[-1]
        last_pid = last_shot["player_id"]
        last_team = "A" if last_pid in ("p1", "p2") else "B"
        end_x, end_y = last_shot["to_court_m"]
        in_width = -0.20 <= end_x <= 6.30
        if last_team == "A":
            landed_opponent_in = in_width and (-0.20 <= end_y <= 6.75)
        else:
            landed_opponent_in = in_width and (6.65 <= end_y <= 13.60)
        winner_team = last_team if landed_opponent_in else ("B" if last_team == "A" else "A")
        if winner_team == "A":
            score_a += 1
        else:
            score_b += 1

        rallies.append(
            {
                "id": rally_id,
                "start_frame": start_f,
                "end_frame": end_f,
                "shot_count": len(rally_shots),
                "winner_team": winner_team,
                "score_after": [score_a, score_b],
            }
        )

    return shots, rallies


def evaluate_shots(pred_shots: List[Dict[str, Any]], gt_shots: List[Dict[str, Any]], tol_frames: int = 5) -> Dict[str, float]:
    if not gt_shots:
        return {"precision": 1.0, "recall": 1.0, "mean_speed_diff_kmh": 0.0}
    matched_gt = set()
    speed_diffs: List[float] = []
    tp = 0
    for ps in pred_shots:
        for g_idx, gs in enumerate(gt_shots):
            if g_idx in matched_gt:
                continue
            if abs(ps["frame"] - gs["frame"]) <= tol_frames:
                matched_gt.add(g_idx)
                tp += 1
                if ps.get("speed_kmh") is not None and gs.get("gt_speed_kmh") is not None:
                    speed_diffs.append(abs(ps["speed_kmh"] - gs["gt_speed_kmh"]))
                break

    precision = round(tp / max(1, len(pred_shots)), 4)
    recall = round(tp / max(1, len(gt_shots)), 4)
    mean_spd_diff = round(sum(speed_diffs) / max(1, len(speed_diffs)), 2) if speed_diffs else 0.0
    return {"precision": precision, "recall": recall, "mean_speed_diff_kmh": mean_spd_diff}


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect badminton shots and estimate shuttle speeds.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights-dir", default=None)
    args = parser.parse_args()

    try:
        vpath = pathlib.Path(args.video)
        calib_data = json.loads(pathlib.Path(args.calib).read_text(encoding="utf-8"))
        court_meta = calibrate_court(calib_data["calib_points"], mode=calib_data.get("mode", "doubles"))
        H = np.array(court_meta["homography_img_to_court"], dtype=np.float64).reshape(3, 3)
        wdir = pathlib.Path(args.weights_dir) if args.weights_dir else None

        shuttle_res = track_shuttle_in_video(vpath, calib_data, weights_dir=wdir)
        players_res = track_players_in_video(vpath, calib_data, weights_dir=wdir)
    except (FileNotFoundError, NotImplementedError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    shots, rallies = detect_shots_and_rallies(
        shuttle_res["frames"], players_res["frames"], H, fps=shuttle_res["fps"]
    )
    metrics = evaluate_shots(shots, calib_data.get("gt_shots", []))

    out_doc = {"video": args.video, "rallies": rallies, "shots": shots, "metrics": metrics}
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Shots detected for {args.video}: P={metrics['precision']} R={metrics['recall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
