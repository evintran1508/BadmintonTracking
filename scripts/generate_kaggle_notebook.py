import json
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "badminton_ai_kaggle.zip"
b64_zip = base64.b64encode(ZIP_PATH.read_bytes()).decode("ascii")

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏸 Badminton AI Studio trên Kaggle GPU (Tesla T4)\n",
                "Chạy toàn bộ hệ thống Web + AI Tracking (YOLO11n-Pose + TrackNetV3) trên GPU miễn phí của Kaggle.\n\n",
                "### ⚙️ Hướng dẫn cài đặt cấu hình Kaggle (Rất quan trọng):\n",
                "1. Ở menu bên phải Kaggle (Notebook Options):\n",
                "   - **Accelerator**: Chọn **GPU T4 x2** (hoặc GPU P100)\n",
                "   - **Internet**: Chọn **Internet On** (bắt buộc bật để tải model và mở link web)\n",
                "2. Bấm nút **Run All** (hoặc Shift+Enter chạy từng cell).\n",
                "3. Nhấp vào đường link `https://xxx.trycloudflare.com` ở Cell cuối cùng để mở Web AI Studio và dùng trực tiếp trên trình duyệt!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Cài đặt các thư viện cần thiết\n",
                "!pip install -q ultralytics fastapi uvicorn python-multipart pycloudflared pyyaml scipy pydantic\n",
                "!apt-get update -qq && apt-get install -y -qq nodejs npm ffmpeg wget\n",
                "print('✅ Cài đặt môi trường hoàn tất!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Tải trọng số mô hình AI (YOLO11n-Pose + TrackNetV3)\n",
                "import os\n",
                "os.makedirs('weights', exist_ok=True)\n",
                "!wget -q -nc -O weights/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt\n",
                "!wget -q -nc -O weights/tracknetv3.pt https://huggingface.co/rahulanantharama/TrackNetV3-badminton/resolve/main/TrackNet_best.pt\n",
                "print('✅ Đã tải xong trọng số AI vào thư mục weights/')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Giải nén mã nguồn dự án Badminton AI Studio\n",
                "import base64, io, zipfile\n",
                f"B64_DATA = '''{b64_zip}'''\n",
                "with zipfile.ZipFile(io.BytesIO(base64.b64decode(B64_DATA))) as zf:\n",
                "    zf.extractall('.')\n",
                "print('✅ Đã giải nén mã nguồn dự án hoàn tất!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Khởi chạy Web Server & Mở đường link Cloudflare Tunnel ra internet\n",
                "import subprocess, time, sys\n",
                "\n",
                "# Tự động cài đặt nếu phiên Kaggle vừa bị reset (tránh lỗi No module named 'pycloudflared')\n",
                "try:\n",
                "    from pycloudflared import try_cloudflare\n",
                "except ImportError:\n",
                "    print('🔄 Đang tự động cài bổ sung pycloudflared & thư viện...')\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycloudflared', 'ultralytics', 'pydantic', 'pyyaml', 'scipy'])\n",
                "    from pycloudflared import try_cloudflare\n",
                "\n",
                "# Kiểm tra Node.js\n",
                "try:\n",
                "    subprocess.run(['node', '-v'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n",
                "except Exception:\n",
                "    print('🔄 Đang cài đặt Node.js...')\n",
                "    subprocess.run('apt-get update -qq && apt-get install -y -qq nodejs npm ffmpeg', shell=True)\n",
                "\n",
                "# Chạy server web Node.js ở cổng 3000\n",
                "p = subprocess.Popen(['node', 'frontend/server.mjs'])\n",
                "time.sleep(3)\n",
                "\n",
                "# Mở Cloudflare Tunnel kết nối tới cổng 3000 (Miễn phí, không cần đăng ký)\n",
                "print('⏳ Đang tạo đường link Cloudflare Tunnel...')\n",
                "tunnel = try_cloudflare(port=3000)\n",
                "\n",
                "print('\\n' + '='*70)\n",
                "print('🏸 BADMINTON AI STUDIO ĐANG CHẠY TRÊN KAGGLE GPU!')\n",
                "print(f'👉 HÃY CLICK VÀO LINK NÀY ĐỂ MỞ WEB: {tunnel.tunnel}')\n",
                "print('='*70 + '\\n')\n",
                "\n",
                "# Giữ tiến trình chạy để web hoạt động\n",
                "try:\n",
                "    p.wait()\n",
                "except KeyboardInterrupt:\n",
                "    p.terminate()\n",
                "    print('Đã dừng server.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_nb = ROOT / "kaggle_badminton_studio.ipynb"
out_nb.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Created {out_nb} ({out_nb.stat().st_size / 1024:.1f} KB)")
