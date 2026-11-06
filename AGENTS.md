# AGENTS.md

## Mục tiêu
Web xem trận cầu lông có HUD đồng bộ theo frame (vòng dưới chân, thẻ tên, LOCK ON, COURT MAP, MOVE).
AI xuất một file JSON theo schema/match.schema.json; web chỉ đọc file đó.

## Lệnh
- Cài đặt: `make setup`        - Trọng số model: `make weights` (không commit)
- Test: `make test`           - Lint: `make lint`
- Chạy AI: `python -m ai_core.pipeline --video <mp4> --calib <json> --out <json>`
- Đánh giá: `python -m eval.run --split dev`
- Web: `cd frontend && MOCK=1 npm run dev`  (MOCK=1 đọc fixtures/mock_match.json)

## Quy tắc
1. Schema là nguồn sự thật. Đổi schema thì đổi luôn mock, fixtures và kiểu TS.
2. Đơn vị: _px, _m, _kmh, _mps. Tọa độ sân: gốc phía xa trái, Y tăng về phía camera, lưới Y=6.7.
3. Mỗi task nhỏ hơn ~300 dòng, có test và dòng Kiểm chứng.
4. Không sửa fixtures/clips/real_test_* và nhãn test; chỉnh ngưỡng chỉ trên real_dev_*.
5. Chưa biết thì ghi docs/OPEN_QUESTIONS.md, dùng mặc định trong config/*.yaml.
6. Thay đổi giao diện phải kèm ảnh chụp Playwright (viewport 390x844).
7. Sau mỗi task cập nhật docs/STATUS.md: làm gì, kiểm chứng ra sao, còn gì.
8. CẤM tự sinh clip/nhãn/số liệu để qua cổng nghiệm thu. Clip giả chỉ được đặt tên synthetic_* và chỉ dùng cho test đơn vị.
9. CẤM đường lui âm thầm (bộ lọc màu, hệ số nhân tốc độ, kẹp giá trị). Không đo được thì trả null; thiếu trọng số thì báo lỗi rõ.
10. STATUS.md ghi mỗi con số kèm loại dữ liệu: "clip thật" hoặc "clip giả". Chỉ số trên clip giả không tính là đạt.
11. Thiếu clip thật, nhãn tay hoặc trọng số: dừng và hỏi người dùng.

## Cấu trúc
ai_core/ (court, players, shuttle, shots, metrics, pipeline)
backend/ (FastAPI, SQLite, worker)
frontend/ là JS thuần + server.mjs. Tách app.js thành module.
fixtures/reference/ = video mẫu + 12 ảnh tham chiếu; fixtures/reference_hud.json = số liệu đọc từ video mẫu.
schema/  config/  fixtures/  eval/  docs/
