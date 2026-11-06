from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ai_core.players import _ensure_torch_numpy_bridge

TRACKNET_WEIGHTS_FILENAME = "tracknetv3.pt"
TRACKNET_H = 288
TRACKNET_W = 512
_TRACKNET_CACHE: Dict[str, Any] = {}


def require_tracknet_weights(weights_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Verify TrackNetV3 weights exist; fail loudly when missing (Rule 9: no silent fallback)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    wdir = weights_dir if weights_dir is not None else (root / "weights")
    weights_file = wdir / TRACKNET_WEIGHTS_FILENAME
    if not weights_file.exists():
        raise FileNotFoundError(f"Thiếu weights/{TRACKNET_WEIGHTS_FILENAME}, chạy make weights")
    return weights_file


def load_tracknet_model(weights_dir: Optional[pathlib.Path] = None) -> Tuple[Any, str]:
    """Load real TrackNetV3 (`seq_len=8`, `bg_mode='concat'`) with `strict=True` checkpoint verification."""
    weights_file = require_tracknet_weights(weights_dir)
    key = str(weights_file.resolve())
    if key in _TRACKNET_CACHE:
        return _TRACKNET_CACHE[key]

    _ensure_torch_numpy_bridge()
    import torch
    import torch.nn as nn

    class Conv2DBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_dim)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.bn(self.conv(x)))

    class Double2DConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.conv_1 = Conv2DBlock(in_dim, out_dim)
            self.conv_2 = Conv2DBlock(out_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv_2(self.conv_1(x))

    class Triple2DConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.conv_1 = Conv2DBlock(in_dim, out_dim)
            self.conv_2 = Conv2DBlock(out_dim, out_dim)
            self.conv_3 = Conv2DBlock(out_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNetV3(nn.Module):
        def __init__(self, in_dim: int = 27, out_dim: int = 8) -> None:
            super().__init__()
            self.down_block_1 = Double2DConv(in_dim, 64)
            self.down_block_2 = Double2DConv(64, 128)
            self.down_block_3 = Triple2DConv(128, 256)
            self.bottleneck = Triple2DConv(256, 512)
            self.up_block_1 = Triple2DConv(768, 256)
            self.up_block_2 = Double2DConv(384, 128)
            self.up_block_3 = Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, kernel_size=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.down_block_1(x)
            x2 = self.down_block_2(self.pool(x1))
            x3 = self.down_block_3(self.pool(x2))
            xb = self.bottleneck(self.pool(x3))
            u1 = self.up_block_1(torch.cat([self.upsample(xb), x3], dim=1))
            u2 = self.up_block_2(torch.cat([self.upsample(u1), x2], dim=1))
            u3 = self.up_block_3(torch.cat([self.upsample(u2), x1], dim=1))
            return torch.sigmoid(self.predictor(u3))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(key, map_location=device, weights_only=False)
    model = TrackNetV3(in_dim=27, out_dim=8).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    _TRACKNET_CACHE[key] = (model, device)
    return model, device


def _extract_heatmap_peak(
    hmap: np.ndarray,
    orig_w: int,
    orig_h: int,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    max_prob = float(np.max(hmap))
    if max_prob < threshold:
        return {"x_px": None, "y_px": None, "visible": False, "conf": round(max_prob, 3)}

    binary = (hmap >= threshold).astype(np.uint8) * 255
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return {"x_px": None, "y_px": None, "visible": False, "conf": 0.0}

    best_lbl = max(range(1, num_labels), key=lambda k: int(stats[k, cv2.CC_STAT_AREA]))
    cx, cy = float(centroids[best_lbl][0]), float(centroids[best_lbl][1])
    x_px = round(cx * orig_w / TRACKNET_W, 2)
    y_px = round(cy * orig_h / TRACKNET_H, 2)
    return {"x_px": x_px, "y_px": y_px, "visible": True, "conf": round(max_prob, 3)}


def _run_tracknetv3_sequence(
    frames_rgb_resized: List[np.ndarray],
    orig_w: int,
    orig_h: int,
    weights_dir: Optional[pathlib.Path] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    import torch

    model, device = load_tracknet_model(weights_dir)
    n = len(frames_rgb_resized)
    if n == 0:
        return []

    # Median background image (3, 288, 512) in [0, 1]
    sample_step = max(1, n // 40)
    sampled = np.stack(frames_rgb_resized[::sample_step], axis=0)
    bg_chw = (np.median(sampled, axis=0).astype(np.float32) / 255.0).transpose(2, 0, 1)

    raw_out: List[Dict[str, Any]] = []
    seq_len = 8
    with torch.no_grad():
        for start in range(0, n, seq_len):
            chunk = frames_rgb_resized[start : start + seq_len]
            actual_len = len(chunk)
            while len(chunk) < seq_len:
                chunk.append(chunk[-1])

            chw_list = [bg_chw] + [(f.astype(np.float32) / 255.0).transpose(2, 0, 1) for f in chunk]
            inp_np = np.concatenate(chw_list, axis=0)[None, ...]  # (1, 27, 288, 512)
            inp_t = torch.from_numpy(inp_np).to(device)
            preds_np = model(inp_t).detach().cpu().numpy()[0]  # (8, 288, 512)

            for k in range(actual_len):
                det = _extract_heatmap_peak(preds_np[k], orig_w, orig_h)
                det["i"] = start + k
                raw_out.append(det)
            if progress_cb is not None:
                progress_cb(min(n, start + actual_len), n)
    return raw_out


def clean_shuttle_trajectory(
    raw_frames: List[Dict[str, Any]],
    fps: float = 30.0,
    max_step_px: float = 85.0,
    max_gap_sec: float = 0.3,
) -> List[Dict[str, Any]]:
    cleaned = [dict(f) for f in raw_frames]
    n = len(cleaned)
    if n == 0:
        return cleaned

    for i in range(1, n - 1):
        cur = cleaned[i]
        prev_f = cleaned[i - 1]
        next_f = cleaned[i + 1]
        if cur["visible"] and prev_f["visible"] and next_f["visible"]:
            d_prev = math.hypot(cur["x_px"] - prev_f["x_px"], cur["y_px"] - prev_f["y_px"])
            d_next = math.hypot(next_f["x_px"] - cur["x_px"], next_f["y_px"] - cur["y_px"])
            d_bridge = math.hypot(next_f["x_px"] - prev_f["x_px"], next_f["y_px"] - prev_f["y_px"])
            if d_prev > max_step_px and d_next > max_step_px and d_bridge <= max_step_px * 2:
                cleaned[i] = {"i": cur.get("i", i), "x_px": None, "y_px": None, "visible": False, "conf": 0.0}

    max_gap_frames = int(round(max_gap_sec * fps))
    vis_indices = [idx for idx, f in enumerate(cleaned) if f["visible"] and f["x_px"] is not None]
    for k in range(len(vis_indices) - 1):
        a = vis_indices[k]
        b = vis_indices[k + 1]
        gap = b - a - 1
        if 1 <= gap <= max_gap_frames:
            xa, ya = cleaned[a]["x_px"], cleaned[a]["y_px"]
            xb, yb = cleaned[b]["x_px"], cleaned[b]["y_px"]
            if math.hypot(xb - xa, yb - ya) <= max_step_px * (gap + 1):
                for m in range(a + 1, b):
                    alpha = (m - a) / (b - a)
                    cleaned[m] = {
                        "i": cleaned[m].get("i", m),
                        "x_px": round(xa * (1.0 - alpha) + xb * alpha, 2),
                        "y_px": round(ya * (1.0 - alpha) + yb * alpha, 2),
                        "visible": True,
                        "conf": 0.55,
                    }

    vx, vy = 0.0, 0.0
    for i in range(1, n):
        if cleaned[i]["visible"] and cleaned[i - 1]["visible"]:
            px, py = cleaned[i - 1]["x_px"], cleaned[i - 1]["y_px"]
            pred_x, pred_y = px + vx, py + vy
            obs_x, obs_y = cleaned[i]["x_px"], cleaned[i]["y_px"]
            kx = 0.75 * obs_x + 0.25 * pred_x
            ky = 0.75 * obs_y + 0.25 * pred_y
            dx, dy = kx - px, ky - py
            step = math.hypot(dx, dy)
            if step > max_step_px:
                scale = max_step_px / step
                dx *= scale
                dy *= scale
            cleaned[i]["x_px"] = round(px + dx, 2)
            cleaned[i]["y_px"] = round(py + dy, 2)
            vx, vy = 0.5 * vx + 0.5 * dx, 0.5 * vy + 0.5 * dy
        else:
            vx, vy = 0.0, 0.0

    return cleaned


def track_shuttle_in_video(
    video_path: pathlib.Path,
    labels_data: Optional[Dict[str, Any]] = None,
    debug_video_path: Optional[pathlib.Path] = None,
    detector_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    weights_dir: Optional[pathlib.Path] = None,
    max_frames: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    if detector_fn is None:
        require_tracknet_weights(weights_dir)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_frames: List[Dict[str, Any]] = []
    frames_bgr: List[np.ndarray] = []
    resized_rgb: List[np.ndarray] = []
    prev_bgr: Optional[np.ndarray] = None
    idx = 0

    limit_frames = max_frames if (max_frames is not None and max_frames > 0) else float("inf")
    target_shuttle_n = int(limit_frames) if limit_frames != float("inf") else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)
    target_shuttle_n = max(1, target_shuttle_n)

    while idx < limit_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if detector_fn is not None:
            det = detector_fn(frame, prev_bgr)
            det["i"] = idx
            raw_frames.append(det)
            prev_bgr = frame
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_rgb.append(cv2.resize(rgb, (TRACKNET_W, TRACKNET_H), interpolation=cv2.INTER_AREA))
            if idx % 30 == 0:
                print(f"STATUS:Tiền xử lý khung hình [TrackNetV3]: Khung hình {idx}/{target_shuttle_n}...", flush=True)
        if debug_video_path is not None:
            frames_bgr.append(frame)
        idx += 1
    cap.release()

    if detector_fn is None:
        raw_frames = _run_tracknetv3_sequence(
            resized_rgb, vw, vh, weights_dir=weights_dir, progress_cb=progress_cb
        )

    max_step = max(65.0, max(vw, vh) * 0.14)
    cleaned = clean_shuttle_trajectory(raw_frames, fps=fps, max_step_px=max_step)

    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(debug_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
        for i, dbg in enumerate(frames_bgr):
            out = dbg.copy()
            sf = cleaned[i]
            if sf["visible"] and sf["x_px"] is not None:
                sx, sy = int(round(sf["x_px"])), int(round(sf["y_px"]))
                cv2.circle(out, (sx, sy), 9, (0, 230, 255), 2)
            writer.write(out)
        writer.release()

    recall_15px = 0.0
    if labels_data and "gt_shuttle_px" in labels_data:
        gt_vis = [g for g in labels_data["gt_shuttle_px"] if g.get("visible") and g.get("x_px") is not None]
        hits = 0
        for g in gt_vis:
            f_i = g["frame"]
            if f_i < len(cleaned) and cleaned[f_i]["visible"] and cleaned[f_i]["x_px"] is not None:
                err = math.hypot(cleaned[f_i]["x_px"] - g["x_px"], cleaned[f_i]["y_px"] - g["y_px"])
                if err <= 15.0:
                    hits += 1
        recall_15px = round(hits / max(1, len(gt_vis)), 4)

    return {
        "video": str(video_path),
        "fps": fps,
        "total_frames": len(cleaned),
        "recall_15px": recall_15px,
        "frames": cleaned,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track shuttlecock with TrackNetV3 and clean trajectory.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug-video", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    try:
        lbl = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8")) if args.labels else None
        dbg = pathlib.Path(args.debug_video) if args.debug_video else None
        wdir = pathlib.Path(args.weights_dir) if args.weights_dir else None
        res = track_shuttle_in_video(
            pathlib.Path(args.video), lbl, dbg, weights_dir=wdir, max_frames=args.max_frames
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"Shuttle tracked {args.video}: recall_15px={res['recall_15px']*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
