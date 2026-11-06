# Trạng thái dự án (STATUS) — Kế hoạch v3

> Quy tắc nghiệm thu v3: Mọi con số đều phải gắn nhãn **`(clip thật)`** hoặc **`(clip giả)`**. Chỉ số trên **`(clip giả)`** chỉ dùng cho test hồi quy đơn vị, **KHÔNG tính là đạt cổng nghiệm thu**.

## Milestone R0: Dọn dẹp và đầu vào thật (HOÀN TẤT 4.5/5 TASK — DỪNG TẠI R0.5 CHỜ 5 CLIP THẬT)

- [x] **Task R0.1**: Đổi tên toàn bộ clip/nhãn giả sang `synthetic_*` (`synthetic_dev_01..03.mp4`, `synthetic_test_01..02.mp4`, `synthetic_walk_test.mp4`); sửa [`eval/run.py`](file:///d:/BadmintonTracking/eval/run.py) in tách biệt hai bảng `synthetic` (hồi quy đơn vị, không tính cổng) và `real` (cổng nghiệm thu chính thức, trả mã lỗi `1` khi thiếu `real_dev_*`).
  - **Kiểm chứng**: `python -m unittest tests/test_r0_1_synthetic_and_eval.py -v` (`OK`); `python -m eval.run --split dev` in đủ 2 bảng (`synthetic` & `real`), bảng `real` báo `BLOCKED` `(clip thật: chưa có dữ liệu)`.
- [x] **Task R0.2**: Xóa `ai_core/onnx_export.py` và `out/shuttle_head.onnx`; chuyển bộ lọc màu sang [`tests/synthetic_helpers.py`](file:///d:/BadmintonTracking/tests/synthetic_helpers.py); gỡ sạch bộ lọc màu dự phòng khỏi [`players.py`](file:///d:/BadmintonTracking/ai_core/players.py), [`shuttle.py`](file:///d:/BadmintonTracking/ai_core/shuttle.py), [`pipeline.py`](file:///d:/BadmintonTracking/ai_core/pipeline.py) và gỡ hệ số nhân `2.1` + kẹp giá trị tốc độ khỏi [`shots.py`](file:///d:/BadmintonTracking/ai_core/shots.py).
  - **Kiểm chứng**: `python -m unittest tests/test_r0_2_no_silent_fallback.py -v` (`OK`) — chạy `python -m ai_core.pipeline` khi thiếu trọng số thoát mã `2 (≠ 0)` kèm thông báo `"Thiếu weights/..., chạy make weights"`.
- [x] **Task R0.3**: Cài đặt `.\make setup` ([`scripts/setup_env.py`](file:///d:/BadmintonTracking/scripts/setup_env.py): cài `ultralytics>=8.3.0`, `torch`, kiểm/tải `ffmpeg` + `ffprobe` có `libx264`) và `.\make weights` ([`scripts/download_weights.py`](file:///d:/BadmintonTracking/scripts/download_weights.py): tải `weights/yolo11n-pose.pt` & `weights/tracknetv3.pt` và kiểm mã băm SHA-256).
  - **Kiểm chứng**: `python -c "from ultralytics import YOLO; YOLO('weights/yolo11n-pose.pt')"` chạy thành công (`exit 0`); `python -m unittest tests/test_r0_3_weights_and_setup.py -v` (`OK`).
- [x] **Task R0.4**: Cập nhật [`docs/PLAN.md`](file:///d:/BadmintonTracking/docs/PLAN.md) sang bản v3 và sửa [`AGENTS.md`](file:///d:/BadmintonTracking/AGENTS.md) (ghi đúng frontend là JS thuần + `server.mjs`, thêm quy tắc 8–11 cấm số liệu tự sinh và đường lui âm thầm).
  - **Kiểm chứng**: `test_agents_md_updated_for_v3` (`OK`).
- [ ] **Task R0.5 (Đã xong phần video mẫu tham chiếu — Đang chờ 5 clip thật không HUD)**:
  - [x] Đã lưu video mẫu vào `fixtures/reference/reference_match.mp4`, trích xuất đủ 12 ảnh theo giây `fixtures/reference/sec_00.jpg` → `sec_11.jpg`, và dựng [`fixtures/reference_hud.json`](file:///d:/BadmintonTracking/fixtures/reference_hud.json) khớp 12 mốc giây ở Bảng mục 4 (qua `python -m schema.validate fixtures/reference_hud.json` và `tests/test_r0_5_reference_hud.py` `OK`).
  - [ ] **ĐANG CHỜ NGƯỜI DÙNG**: Copy 5 clip trận đấu thật **không có HUD** (`real_dev_01..03`, `real_test_01..02`) vào `fixtures/clips/` (và `real_walk_test` vào `fixtures/` khi sẵn sàng).

## Tích hợp chạy thật đầu-cuối cho người dùng test trực tiếp trên Web
- [x] Đã nối mô hình thật `YOLO11n-Pose` ([`ai_core/players.py`](file:///d:/BadmintonTracking/ai_core/players.py)) + `TrackNetV3` `strict=True` ([`ai_core/shuttle.py`](file:///d:/BadmintonTracking/ai_core/shuttle.py)) + chuẩn hóa H.264/AAC bằng `ffmpeg` ([`ai_core/video_norm.py`](file:///d:/BadmintonTracking/ai_core/video_norm.py), [`backend/app.py`](file:///d:/BadmintonTracking/backend/app.py), [`frontend/server.mjs`](file:///d:/BadmintonTracking/frontend/server.mjs)).
- **Kiểm chứng**:
  - Smoke test 16 frame trên video mẫu `reference_match.mp4`: `4/4 người chơi (clip thật - video mẫu có sẵn HUD, chỉ dùng kiểm thử luồng chạy, KHÔNG tính cổng)`, `6/16 frame thấy cầu (clip thật - video mẫu)`, qua `python -m schema.validate` (`OK`).
  - `.\make lint` (`OK`) và `.\make test` (`39/39 tests OK`, bao gồm ảnh chụp Playwright `390x844` trên `clip giả`).
- [x] **Cập nhật giao diện Minimalism + Tính điểm trận đấu (BWF 21) + Kéo trượt tua video mượt + Âm thanh video & Web Audio SFX**:
  - Thêm mô-đun [`frontend/public/audio_score.mjs`](file:///d:/BadmintonTracking/frontend/public/audio_score.mjs) (quản lý tỉ số BWF 21 điểm, lịch sử điểm kèm mốc thời gian để bấm tua lại, hoàn tác/reset, và bộ tổng hợp âm thanh tiếng vợt chạm cầu + tiếng chuông báo điểm bằng Web Audio API).
  - Thiết kế lại [`frontend/public/index.html`](file:///d:/BadmintonTracking/frontend/public/index.html) và [`frontend/public/app.js`](file:///d:/BadmintonTracking/frontend/public/app.js) theo phong cách tối giản (Minimalism), bổ sung khóa `isScrubbing` chống giật thanh trượt khi kéo tua video, nút `-3s`/`+3s`, nút bật/tắt tiếng (`#btn-mute`) + thanh âm lượng (`#volume-slider`) và hiển thị bảng tỉ số trực tiếp trên HUD (`drawn.scoreBoard`).
  - **Kiểm chứng**: `node --test frontend/tests/projection.test.mjs` (`4/4 tests OK (clip giả)`), `python -m unittest tests/test_hud_overlay.py tests/test_ui_calib.py tests/test_summary_ui.py tests/test_web_sync.py -v` (`4/4 tests OK (clip giả)`, chụp lại `out/hud_390x844.png`, `out/calib_ui_390x844.png`, `out/summary_ui_390x844.png` ở viewport `390x844`).

- [x] **Khắc phục lỗi theo dõi người chơi, sân và xử lý toàn bộ video (F0–F3)**:
  - **Dọn sạch dữ liệu bịa & kẹp giá trị (Quy tắc 9)**: Gỡ bỏ việc tự sinh bbox/tọa độ chân giả `[0, 0]` trong [`ai_core/pipeline.py`](file:///d:/BadmintonTracking/ai_core/pipeline.py) khi mất dấu người chơi; nếu không phát hiện được thì trả về `bbox_px: null`, `foot_px: null`, `foot_court_m: null`, `conf: 0.0`. Gỡ kẹp vận tốc cứng trong [`ai_core/metrics.py`](file:///d:/BadmintonTracking/ai_core/metrics.py).
  - **Loại bỏ bộ lọc màu trong định hình sân (Quy tắc 9)**: Gỡ sạch bộ lọc màu xanh lá/xanh dương (`green_mask`, `blue_mask`) trong [`ai_core/court.py`](file:///d:/BadmintonTracking/ai_core/court.py), thiết lập khung 6 mốc chuẩn an toàn nằm hoàn toàn trong khung hình video cho mọi độ phân giải.
  - **Tối ưu nhận diện YOLO11-pose & gán người chơi**: Tự động nâng `imgsz=1280` cho video HD/FullHD trong [`ai_core/players.py`](file:///d:/BadmintonTracking/ai_core/players.py) để nhận diện rõ 2 người chơi ở sân xa; tự động hút vị trí nhấp chuột vào người chơi gần nhất để gán đúng tọa độ chân trên mặt sân.
  - **Hỗ trợ xử lý toàn bộ video**: Thêm tùy chọn "Toàn bộ video" (`max_frames: 0`) trong giao diện [`frontend/public/index.html`](file:///d:/BadmintonTracking/frontend/public/index.html); sửa lỗi JavaScript `|| 180` nuốt giá trị `0` trong [`frontend/server.mjs`](file:///d:/BadmintonTracking/frontend/server.mjs) và [`frontend/public/app.js`](file:///d:/BadmintonTracking/frontend/public/app.js); HUD hiển thị trạng thái rõ ràng khi tua qua phân đoạn chưa xử lý thay vì đóng băng khung hình.
  - **Kiểm chứng**: `make lint` (`exit 0`), `python -m unittest discover tests -v` (`37/37 tests OK (clip giả)`), `node --test frontend/tests/projection.test.mjs` (`4/4 tests OK (clip giả)`).

- [x] **Khắc phục lỗi thanh tiến trình dừng ở 55% & quản lý tác vụ xử lý AI nền**:
  - **Sửa lỗi công thức tính phần trăm trong `ai_core/pipeline.py`**: Khi chọn "Toàn bộ video" (`max_frames: 0`), biểu thức `min(est_total or 300, max_frames)` trả về 0 khiến mẫu số bằng 1, đẩy tiến trình lên 55% ngay tại khung hình 30 và kẹt ở 55% suốt toàn bộ thời gian chạy trên CPU. Đã sửa thành `target_total = int(limit_frames) if limit_frames != float("inf") else (est_total if est_total > 0 else 300)`.
  - **Bổ sung báo cáo tiến độ chi tiết theo từng khung hình**: Phát luồng thông báo `STATUS:Bám người chơi [YOLO11n-Pose]: Khung hình X/Y (P%)` và `STATUS:Dò quỹ đạo cầu [TrackNetV3]: Khung hình X/Y (P%)` trực tiếp từ Python stdout; thêm cập nhật trong bước nạp khung hình TrackNetV3 (`ai_core/shuttle.py`).
  - **Quản lý tiến trình con & hủy tác vụ trong `frontend/server.mjs`**: Lưu `activeChildProcess`, tự động hủy tiến trình cũ bằng `taskkill /PID ... /T /F` khi người dùng bấm xử lý lại để tránh nghẽn 100% CPU do nhiều tiến trình chạy song song; bổ sung endpoint `POST /api/cancel-process` và nút "Hủy" trên giao diện [`frontend/public/index.html`](file:///d:/BadmintonTracking/frontend/public/index.html).
  - **Tối ưu tùy chọn thời lượng mặc định trên giao diện**: Đặt mặc định xem thử 10 giây đầu (`max_frames: 300`, ~30 giây xử lý trên CPU) kèm ước tính thời gian thực tế rõ ràng cho các mức 6s, 10s, 15s, 30s, 1m, 3m, và "Toàn bộ video (~35 phút trên CPU)".
  - **Hỗ trợ định dạng điểm hiệu chỉnh linh hoạt trong `ai_core/court.py`**: Chấp nhận cả khóa `px` và `img_px` tránh lỗi `KeyError`.
  - **Kiểm chứng**:
    - Chạy thử nghiệm thực tế luồng API `/api/process` và `/api/process-status`: tiến độ tăng mượt từ 8% -> 17% -> 25% -> 56% -> 66% -> 93% -> 100% `COMPLETED` với schema JSON hợp lệ `(clip giả)`.
    - `python -m unittest discover tests -v` (`39/39 tests OK (clip giả)`).
    - `node --test frontend/tests/projection.test.mjs` (`4/4 tests OK (clip giả)`).
    - Chụp ảnh giao diện Playwright `out/setup_390x844.png` và `out/hud_390x844.png` (viewport 390x844).

- [x] **Đóng gói triển khai Kaggle GPU (Tesla T4) chạy Web AI Studio trực tiếp**:
  - Đóng gói toàn bộ mã nguồn sạch thành [`badminton_ai_kaggle.zip`](file:///d:/BadmintonTracking/badminton_ai_kaggle.zip) (63.7 KB) bằng [`scripts/pack_kaggle.py`](file:///d:/BadmintonTracking/scripts/pack_kaggle.py), bao gồm đầy đủ `schema/models.py`, `schema/validate.py` để khắc phục lỗi thiếu `schema.models`.
  - Khắc phục tương thích thiết bị CUDA trong [`ai_core/shuttle.py`](file:///d:/BadmintonTracking/ai_core/shuttle.py) (`preds_np = model(inp_t).detach().cpu().numpy()[0]`) và cấu hình đường dẫn `PYTHON_BIN` môi trường Linux trong [`frontend/server.mjs`](file:///d:/BadmintonTracking/frontend/server.mjs).
  - Tự động sinh file Kaggle Notebook hoàn chỉnh [`kaggle_badminton_studio.ipynb`](file:///d:/BadmintonTracking/kaggle_badminton_studio.ipynb): nạp sẵn mã nguồn, tự động tải trọng số YOLO11n + TrackNetV3, khởi chạy web server và mở Cloudflare Tunnel (`try_cloudflare`) cho ra link web công khai sử dụng GPU T4 miễn phí.
  - **Kiểm chứng**: `python scripts/pack_kaggle.py` (`OK`), `python scripts/generate_kaggle_notebook.py` (`OK`), `python -m unittest discover tests -v` (`39/39 tests OK (clip giả)`).

- [x] **Nâng cấp v4: Tự động canh sân (White-Line Top-Hat + Hough), Tự động chọn 4 người chơi trong sân, Phân loại cú đánh, Tự động tính điểm BWF & Giao diện phân tích lối chơi trực quan**:
  - **Tự động canh 6 điểm vạch sân (`ai_core/court.py`)**: Thay thế bộ lọc màu xanh bằng thuật toán 2 giai đoạn dùng Morphological Top-Hat lọc vạch trắng + giao điểm đường thẳng `cv2.HoughLinesP`, hoạt động trên cả sân thẳng đứng (`synthetic_dev_01..03.mp4`, `fallback_manual: False`, `reproj_error_m <= 0.0001m (clip giả)`) và sân xám góc chéo (`out/uploaded_input.mp4`, `fallback_manual: False (clip thật - video tải lên)`).
  - **Tự động nhận diện & lọc đúng 4 người chơi trong sân (`ai_core/players.py`, `ai_core/pipeline.py`, `frontend/server.mjs`)**: Bổ sung `_court_interior_score`, `select_active_court_players` và `select_best_seed_frame_dets` quét nhiều khung hình đầu (`0, 15, 30, 50`), ưu tiên người đứng trong lòng sân ($0 \le X \le W, 0 \le Y \le L$) chia đều sân gần/sân xa và loại bỏ hoàn toàn khán giả ngồi ghế biên (`X < 0`); thêm API `POST /api/auto-setup` và nút `"⚡ Tự động Canh Sân & Người"`.
  - **Đếm chạm cầu chính xác, phân loại cú đánh & tự động chấm điểm rally (`ai_core/shots.py`, `schema/models.py`, `schema/match.schema.json`)**: Loại bỏ cú chạm cầu ảo tại đỉnh parabol khi cầu bay bổng xa người chơi (`d_court > 2.8m` & `d_px > 145px`); phân loại cú đánh (`shot_type`: `Smash`, `Drive`, `Clear`, `Drop`, `Serve`); tự động tách `rallies` khi cầu nghỉ $> 1.6\text{s}$ và suy luận đội ăn điểm (`winner_team: "A" | "B"`, `score_after: [scoreA, scoreB]`) đồng bộ trực tiếp lên bảng điểm HUD theo thời gian thực.
  - **Giao diện Phân tích lối chơi & Cú đánh tốt nhất (`frontend/public/index.html`, `frontend/public/app.js`)**: Thêm thẻ Highlight **🔥 Cú đánh tốc độ tốt nhất** kèm nút `"▶ Xem pha cầu này"`, thẻ **Phân tích lối chơi & Phong độ** từng người (nhãn phong cách chơi tự động, số lần chạm cầu, tốc độ cầu tối đa `km/h`, quãng đường `m`, tốc độ chạy tối đa `km/h`, thanh 3 màu phân bổ vị trí *Trên lưới / Giữa sân / Cuối sân*), và cập nhật gói Kaggle GPU (`badminton_ai_kaggle.zip`, `kaggle_badminton_studio.ipynb`).
  - **Kiểm chứng**:
    - `.\make.bat lint` (`exit 0`) & `python -m schema.validate --export` (`OK`).
    - `python -m unittest discover -s tests` (`39/39 tests OK (clip giả)`).
    - `python -m unittest discover -s tests -p "test_*ui*.py" -v` (`2/2 Playwright UI tests OK (clip giả)`, chụp `out/summary_ui_390x844.png` và `out/calib_ui_390x844.png` ở viewport `390x844`).
    - Kiểm thử trên video tải lên `out/uploaded_input.mp4`: Tự động canh 6 điểm sân (`fallback_manual: False`) và chọn đúng 4 vận động viên trên sân `[('p1', 'A', [0.658, 9.599]), ('p2', 'A', [3.29, 10.549]), ('p3', 'B', [1.454, 0.139]), ('p4', 'B', [1.535, 3.543])] (clip thật - video người dùng tải lên)`.

## Kết quả `python -m eval.run --split dev` hiện tại

- **Bảng 1 (`synthetic` — Hồi quy đơn vị, KHÔNG tính đạt cổng nghiệm thu)**:
  - Sân: `0.000 m (clip giả)` | Người chơi: `100.0% & 0 lần (clip giả)` | Quãng đường: `13.53 m / 0.97% (clip giả)` | Quả cầu: `100.0% (clip giả)` | Cú đánh: `P=1.00, R=1.00 (clip giả)` | Đồng bộ Web: `0 frame (clip giả)`.
- **Bảng 2 (`real` — Cổng nghiệm thu chính thức)**:
  - Toàn bộ 7 mô-đun: **`BLOCKED (clip thật: thiếu real_dev_01, real_dev_02, real_dev_03)`**.
