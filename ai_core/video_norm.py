from __future__ import annotations
import json
import os
import pathlib
import shutil
import subprocess
from typing import Any, Dict, Tuple


def resolve_ffmpeg_binaries() -> Tuple[str, str]:
    """Resolve `ffmpeg` and `ffprobe` from system PATH or project-local `tools/ffmpeg/`."""
    root = pathlib.Path(__file__).resolve().parent.parent
    exe_suffix = ".exe" if os.name == "nt" else ""

    sys_ffmpeg = shutil.which("ffmpeg")
    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffmpeg and sys_ffprobe:
        return sys_ffmpeg, sys_ffprobe

    tools_ffmpeg = root / "tools" / "ffmpeg"
    if tools_ffmpeg.exists():
        found_ffmpeg = next(tools_ffmpeg.rglob(f"ffmpeg{exe_suffix}"), None)
        found_ffprobe = next(tools_ffmpeg.rglob(f"ffprobe{exe_suffix}"), None)
        if found_ffmpeg and found_ffprobe:
            return str(found_ffmpeg), str(found_ffprobe)

    raise FileNotFoundError("Thiếu ffmpeg/ffprobe, hãy chạy make setup để cài đặt và kiểm tra ffmpeg.")


def probe_video_stream(in_path: pathlib.Path) -> Dict[str, Any]:
    """Probe duration, dimensions, and codec via `ffprobe`."""
    _ffmpeg_bin, ffprobe_bin = resolve_ffmpeg_binaries()
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,r_frame_rate:format=duration",
            "-of",
            "json",
            str(in_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError("Video tải lên bị hỏng hoặc định dạng không đọc được (MEDIA_ERR)")
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise ValueError("Video tải lên không có luồng hình ảnh hợp lệ")
    st = streams[0]
    width = int(st.get("width") or 0)
    height = int(st.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Video tải lên không có khung hình hợp lệ")
    dur = float((info.get("format") or {}).get("duration") or 0.0)
    return {"width": width, "height": height, "duration_sec": dur, "codec_name": st.get("codec_name", "")}


def normalize_video_h264(
    in_path: pathlib.Path,
    out_mp4_path: pathlib.Path,
    max_duration_sec: float = 120.0,
) -> float:
    """Normalize uploaded video to H.264 + AAC MP4 (`+faststart`) via `ffmpeg`, preserving audio & rotation."""
    meta = probe_video_stream(in_path)
    duration_sec = meta["duration_sec"]
    if duration_sec > max_duration_sec:
        raise PermissionError(f"Video dài {duration_sec:.1f}s vượt giới hạn {max_duration_sec}s")

    ffmpeg_bin, _ffprobe_bin = resolve_ffmpeg_binaries()
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(in_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            str(out_mp4_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_mp4_path.exists():
        raise ValueError(f"Lỗi chuyển mã H.264 bằng ffmpeg: {(proc.stderr or '')[-240:]}")
    return duration_sec

