from __future__ import annotations
import pathlib
from typing import Any, Dict
import cv2
import numpy as np


def export_hud_mp4(
    video_path: pathlib.Path,
    match_doc: Dict[str, Any],
    out_mp4_path: pathlib.Path,
) -> Dict[str, Any]:
    """Render HUD overlay directly onto video frames from match.json with 0-frame lag."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(match_doc["video"]["fps"])
    vw = int(match_doc["video"]["width"])
    vh = int(match_doc["video"]["height"])

    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))

    H = np.array(match_doc["court"]["homography_img_to_court"], dtype=np.float64).reshape(3, 3)
    H_inv = np.linalg.inv(H)

    def c2p(xm: float, ym: float) -> tuple[int, int]:
        v = H_inv @ np.array([xm, ym, 1.0], dtype=np.float64)
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    frames = match_doc["frames"]
    shots = match_doc["shots"]
    rendered = 0

    for idx, f_data in enumerate(frames):
        ok, img = cap.read()
        if not ok:
            break
        # 1. Top bar (Rally time, count, speed)
        cv2.rectangle(img, (8, 8), (vw - 8, 48), (22, 13, 9), -1)
        past_shots = [s for s in shots if s["frame"] <= idx]
        last_spd = past_shots[-1]["speed_kmh"] if past_shots else None
        top_txt = f"TIME {f_data['t']:05.2f}s | SHOTS {len(past_shots)} | SPD {last_spd or '--'} km/h"
        cv2.putText(img, top_txt, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 189, 56), 2)

        # 2. Foot rings + floating tags
        for p in match_doc["players"]:
            pid = p["id"]
            pf = f_data["players"].get(pid)
            if not pf or not pf.get("foot_court_m"):
                continue
            xm, ym = pf["foot_court_m"]
            fx, fy = c2p(xm, ym)
            cv2.ellipse(img, (fx, fy), (22, 9), 0, 0, 360, (238, 211, 34), 2)
            if pf.get("bbox_px"):
                bx1, by1, _bx2, _by2 = [int(v) for v in pf["bbox_px"]]
                cv2.putText(
                    img,
                    f"{p['name']} E:{int(pf['energy'])} M:{pf['dist_cum_m']:.1f}m",
                    (bx1 - 10, max(60, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                )

        # 3. Shuttle LOCK ON reticle
        sf = f_data.get("shuttle", {})
        if sf.get("visible") and sf.get("x_px") is not None:
            sx, sy = int(round(sf["x_px"])), int(round(sf["y_px"]))
            cv2.rectangle(img, (sx - 10, sy - 10), (sx + 10, sy + 10), (21, 204, 250), 2)

        writer.write(img)
        rendered += 1

    cap.release()
    writer.release()
    return {
        "out_mp4": str(out_mp4_path),
        "rendered_frames": rendered,
        "expected_frames": len(frames),
        "frame_lag": abs(rendered - len(frames)),
    }

