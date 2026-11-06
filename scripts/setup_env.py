from __future__ import annotations
import os
import pathlib
import subprocess
import sys
import urllib.request
import zipfile

FFMPEG_WIN64_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip"
)


def ensure_ffmpeg(root: pathlib.Path) -> None:
    from ai_core.video_norm import resolve_ffmpeg_binaries

    try:
        ff, fp = resolve_ffmpeg_binaries()
        subprocess.run([ff, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([fp, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"FFmpeg OK: {ff}")
        return
    except FileNotFoundError:
        if os.name != "nt":
            raise RuntimeError("Thiếu ffmpeg/ffprobe trên hệ thống. Cài đặt bằng apt/brew install ffmpeg.")

    tools_dir = root / "tools" / "ffmpeg"
    tools_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tools_dir / "ffmpeg-win64-gpl.zip"
    print(f"Downloading portable FFmpeg (H.264 + AAC) -> {zip_path} ...")
    urllib.request.urlretrieve(FFMPEG_WIN64_URL, str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tools_dir)
    zip_path.unlink(missing_ok=True)

    ff, fp = resolve_ffmpeg_binaries()
    subprocess.run([ff, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([fp, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Portable FFmpeg installed: {ff}")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "pydantic", "pyyaml", "jsonschema", "torch", "ultralytics>=8.3.0"],
        check=True,
    )
    ensure_ffmpeg(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

