from __future__ import annotations
import hashlib
import pathlib
import subprocess
import shutil
import sys
import urllib.request
from typing import Dict

WEIGHT_SPECS: Dict[str, Dict[str, str]] = {
    "yolo11n-pose.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
        "sha256": "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
    },
    "tracknetv3.pt": {
        "url": "https://huggingface.co/rahulanantharama/TrackNetV3-badminton/resolve/main/TrackNet_best.pt",
        "sha256": "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a",
    },
}


def compute_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: pathlib.Path, expected_sha256: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Thiếu file trọng số: {path}")
    actual = compute_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"Sai mã băm SHA-256 cho {path.name}: expected={expected_sha256}, actual={actual}")
    return actual


def verify_all_weights(weights_dir: pathlib.Path) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for fname, spec in WEIGHT_SPECS.items():
        verify_sha256(weights_dir / fname, spec["sha256"])
        results[fname] = True
    return results


def download_and_verify_weights(weights_dir: pathlib.Path) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")

    for fname, spec in WEIGHT_SPECS.items():
        dest = weights_dir / fname
        if dest.exists():
            try:
                verify_sha256(dest, spec["sha256"])
                print(f"[OK] {fname} đã có sẵn và khớp SHA-256 ({spec['sha256'][:12]}...)")
                continue
            except ValueError:
                dest.unlink(missing_ok=True)

        print(f"Đang tải {fname} từ {spec['url']} ...")
        tmp_dest = weights_dir / f"{fname}.part"
        tmp_dest.unlink(missing_ok=True)
        if curl_bin:
            subprocess.run([curl_bin, "-L", "--fail", "-o", str(tmp_dest), spec["url"]], check=True)
        else:
            urllib.request.urlretrieve(spec["url"], str(tmp_dest))

        try:
            verify_sha256(tmp_dest, spec["sha256"])
        except Exception:
            tmp_dest.unlink(missing_ok=True)
            raise
        tmp_dest.replace(dest)
        print(f"[OK] Đã tải & kiểm SHA-256 thành công: {fname} ({spec['sha256'][:12]}...)")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    download_and_verify_weights(root / "weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())

