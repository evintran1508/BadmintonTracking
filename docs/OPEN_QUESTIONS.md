# Open Questions & Decisions

1. **Công thức chỉ số năng lượng (`energy`)**: Video mẫu giảm từ 94 xuống 61 trong 11 giây. Cấu hình mặc định trong `config/metrics.yaml` (`initial: 100`, `dist_decay_per_m: 1.2`, `shot_decay: 0.8`).
2. **Ranh giới vùng sân (`front` / `mid` / `back`)**: Sân dài 13.4m (mỗi nửa sân 6.7m tính từ lưới `Y = 6.7`). Chia theo khoảng cách tới lưới trong `config/metrics.yaml`: `front` (0–1.98m), `mid` (1.98–4.72m), `back` (4.72–6.70m).
3. **Bán kính vòng elip dưới chân người chơi**: Mặc định `foot_ring_radius_m: 0.45` m trong `config/metrics.yaml`.
4. **[Đã chốt R0.3] Phần cứng & `ffmpeg`**: Máy dùng `Intel(R) Graphics` (chạy CPU), đã cài bản portable `ffmpeg` + `ffprobe` (`libx264` + `aac`) vào `tools/ffmpeg/` qua `make setup`. Ngưỡng tốc độ xử lý đặt riêng cho CPU trong `config/metrics.yaml` (`max_processing_ratio_cpu: 8.0`, GPU `2.0`).
5. **[Đã chốt R0.3] Giấy phép mô hình**: Dự án dùng mục đích cá nhân / nghiên cứu → sử dụng chính thức `weights/yolo11n-pose.pt` (AGPL-3.0) và `weights/tracknetv3.pt` (TrackNetV3 badminton checkpoint), kiểm chứng SHA-256 trong `make weights`.
6. **[Đang chờ người dùng — R0.5] Đầu vào video thật**: Đã hoàn tất phần trích 12 ảnh tham chiếu (`fixtures/reference/sec_00.jpg`..`sec_11.jpg`) và `fixtures/reference_hud.json` từ video mẫu. Hiện dừng chờ người dùng copy 5 clip thật không có HUD (`real_dev_01..03`, `real_test_01..02`) vào `fixtures/clips/` và clip `real_walk_test` vào `fixtures/`.
