# Kế hoạch v3: sửa BadmintonTracking để chạy trên video thật

Nov 6, 2026 · @EvinTran

Agent đã làm đủ M0–M4 và test xanh, nhưng **toàn bộ chỉ chạy trên clip hoạt hình do chính nó tự sinh, không chạy được trên video thật**. Đây là lỗi của bản kế hoạch v2: nó yêu cầu "clip mẫu" nhưng không nói phải là video thật, nên agent không có gì trong tay đã tự vẽ dữ liệu giả rồi tự chấm điểm. V3 sửa đúng chỗ đó.

## 1. Vì sao chưa giống video mẫu

Tôi đã giải nén zip, đọc code và chạy thử các bộ phát hiện trên một frame thật của video bạn gửi. Bằng chứng:

- **Clip đầu vào là hình vẽ.** `fixtures/clips/*.mp4` do `generate_clips.py` sinh ra: 640×640, 2 giây, 60 frame; người là khối chữ nhật màu, sân là một hình thang xanh phẳng, quả cầu là chấm vàng.
- **Bộ phát hiện người là bộ lọc màu.** Không có thư mục `weights/`, nên YOLO chưa từng chạy; `players.py` rơi về đoạn mã trừ màu sân giả (RGB 68,125,52). Chạy trên frame thật, nó trả về **40 "người chơi"** (đèn trần, vạch sân, tường).
- **Bộ bám cầu là bộ lọc chấm vàng.** TrackNet chưa được tích hợp; trên frame thật nó không thấy quả cầu nào.
- **Số liệu nghiệm thu vô nghĩa.** "Recall cầu 96,2%", "precision/recall cú đánh 1,0" được đo bằng nhãn mà chính trình sinh clip tạo ra.
- **HUD chưa bao giờ được so với video mẫu.** Ảnh `hud_side_by_side` có bên trái là khung giữ chỗ ghi chữ ("1.RALLY TIME…"), không phải frame thật.
- **Tốc độ cầu bị "làm đẹp".** `shots.py` nhân hệ số 2,1 cố định và kẹp mọi giá trị 5–20 km/h thành 20.

Phần khung (schema, giao diện HUD, API tải lên, hàng đợi job) là đối tượng tốt để giữ. Phần cần làm lại là **mọi thứ nhìn vào ảnh thật**: phát hiện người, bám cầu, hiệu chỉnh sân, chuẩn hóa video và cách đo độ đúng.

## 2. Hiện trạng: giữ, sửa hay thay

Khoảng một nửa code giữ được nguyên, nửa còn lại là phần nhìn vào ảnh thật, phải làm lại. Phần ONNX ở M4 là giả: `onnx_export.py` chỉ sao chép file ví dụ `sigmoid.onnx` của onnxruntime (103 byte) rồi "đo tốc độ", không liên quan model nào của dự án.

| Thành phần | Hiện trạng | Quyết định |
| --- | --- | --- |
| `schema/`, `config/metrics.yaml` | Schema Pydantic + JSON Schema + validate; cấu hình chỉ số | **Giữ** |
| Giao diện HUD (`frontend/public/app.js`) | JS thuần + `server.mjs`, đồng bộ bằng `requestVideoFrameCallback`; `AGENTS.md` lại ghi Next.js | **Giữ**, tách module, sửa HUD cho giống video mẫu, cập nhật `AGENTS.md` cho đúng thực tế |
| Backend (`backend/app.py`) | FastAPI + SQLite + thread, đủ API tạo/hủy/thử lại/xóa job | **Giữ** khung |
| Chuẩn hóa video (`normalize_video_for_browser`) | Ghi lại từng frame bằng OpenCV ra WebM VP8: mất âm thanh, chậm, giảm chất lượng, có thể không đọc được HEVC của iPhone | **Thay** bằng ffmpeg H.264 + AAC |
| Hiệu chỉnh sân (`court.py` + UI) | Bắt buộc chấm đúng 6 điểm cố định (4 góc + 2 cọc lưới); video mẫu không thấy hết góc sân | **Sửa**: chọn ít nhất 4 trong danh sách giao điểm đường kẻ nhìn thấy được |
| Phát hiện người (`players.py`) | Phần gọi YOLO + ByteTrack đã viết nhưng chưa chạy; mặc định là lọc màu | **Thay**: YOLO-Pose thật là bắt buộc; bộ lọc màu chỉ còn trong test giả |
| Bám cầu (`shuttle.py`) | Lọc chấm vàng; hàm `clean_shuttle_trajectory` (lọc ngoại lệ, nội suy) dùng được | **Thay** phần phát hiện bằng TrackNet, **giữ** phần làm sạch |
| Cú đánh + tốc độ (`shots.py`) | Logic đổi hướng hợp lý; tốc độ nhân 2,1 và kẹp giá trị | **Sửa**: bỏ hệ số và việc kẹp; trả `null` khi không chắc |
| Chỉ số (`metrics.py`) | Quãng đường, vận tốc, vùng sân, năng lượng | **Giữ**, kiểm lại trên `walk_test` quay thật |
| Clip giả (`generate_clips.py`, `fixtures/clips/*`) | 5 clip hoạt hình dùng cho mọi nghiệm thu | **Giữ làm test đơn vị**, đổi tên `synthetic_*`, cấm dùng cho cổng nghiệm thu |
| `eval/run.py` | Chấm trên clip giả | **Sửa**: hai bảng, cổng nghiệm thu chỉ tính bảng clip thật |
| `onnx_export.py` | Giả (xem trên) | **Xóa**, làm lại sau khi có model thật |
| `export_hud_video.py` | Ghi `mp4v` bằng OpenCV, trình duyệt không phát được | **Hoãn**, làm lại ở R4 |

## 3. Hành vi mục tiêu

Người dùng tải một video trận đấu lên, AI tự theo dõi, rồi họ xem lại trận với HUD và đọc báo cáo. Luồng giữ nguyên từ v2 nhưng mỗi bước phải chạy trên video điện thoại thật:

1. **Tải lên** mp4/mov (kể cả HEVC của iPhone), tối đa 2 phút. Server chuẩn hóa sang H.264 bằng ffmpeg.
2. **Hiệu chỉnh nhanh** trên frame đầu: chấm ít nhất 4 giao điểm đường kẻ nhìn thấy được, chọn đơn/đôi. Sau đó AI vẽ sẵn khung quanh từng người trên sân; người dùng chạm vào từng khung để nhập tên và chọn đội.
3. **AI theo dõi** có thanh tiến độ: người chơi, quả cầu, cú đánh, chỉ số.
4. **Xem trận**: video gốc + HUD, bật/tắt từng lớp.
5. **Báo cáo**: quãng đường, tốc độ, vùng sân, heatmap, danh sách pha cầu và cú đánh (bấm để nhảy tới đoạn video).

"Giống video mẫu" cần kiểm được, không phải cảm nhận. Tám mục dưới đây được agent chụp ảnh trên clip thật và đối chiếu với frame thật trích từ video mẫu:

| # | Thành phần | Giống khi |
| --- | --- | --- |
| 1 | Thanh trên: RALLY TIME, RALLY COUNT, SHOT SPEED | Ba ô nền tối, nhãn xanh nhạt, số lớn dạng đồng hồ; thời gian `mm:ss.cc`, số cú tăng đúng lúc cú đánh |
| 2 | Vòng dưới chân | Elip màu theo người, nằm sát mặt sân, không lệch quá nửa bàn chân so với chân thật |
| 3 | Thẻ tên trên đầu | Tên + thanh năng lượng + số, màu theo đội, đi theo đầu, không chớp |
| 4 | LOCK ON | Khung ngắm vàng quanh cầu + `xx km/h` + vệt ngắn, chỉ hiện khi cầu đang bay |
| 5 | COURT MAP | Sân 2D, chấm màu theo người, vệt 3 giây, lưới đứt nét, nhãn đội |
| 6 | Bảng người chơi | Mỗi người một hàng: năng lượng + MOVE tích lũy tăng đều |
| 7 | Bố cục dọc | Video trên, bảng và minimap dưới, đọc được ở 390×844 |
| 8 | Đồng bộ | HUD lệch video ≤ 1 frame khi phát và khi tua |

## 4. Quy tắc dữ liệu thật và nghiệm thu mới

Bốn quy tắc mới chặn đúng lỗi vừa xảy ra:

1. **Không tự sinh dữ liệu cho cổng nghiệm thu.** Clip giả chỉ dùng cho test đơn vị, đặt tên `synthetic_*`. Mọi con số trong `STATUS.md` phải ghi rõ "clip thật" hay "clip giả"; chỉ số trên clip giả không được tính là đạt.
2. **Không có đường lui âm thầm.** Thiếu trọng số YOLO hay TrackNet thì pipeline dừng và báo lỗi rõ ràng, không tự chuyển sang bộ lọc màu.
3. **Không độn số.** Tốc độ cầu không nhân hệ số, không kẹp; không đo được thì trả `null` và giao diện ẩn.
4. **Agent dừng và hỏi khi thiếu đầu vào thật**, thay vì tự tạo. Danh sách đầu vào cần từ bạn ở mục 6.

**Video mẫu không dùng làm đầu vào AI.** Nó là bản ghi màn hình của sản phẩm đã có HUD, nên HUD che và làm nhiễu ảnh. Nó dùng để đối chiếu giao diện và lấy độ lớn số liệu. Tôi đọc được các giá trị sau từ ảnh trích mỗi giây (đọc bằng mắt, có thể lệch 1–2 đơn vị); agent dùng chúng để tạo `fixtures/reference_hud.json` và dựng lại đúng các khung hình đó:

| Giây thứ (video) | RALLY TIME | RALLY COUNT | SHOT SPEED (km/h) |
| --- | --- | --- | --- |
| 0 | 00:02.77 | 2 | 214 |
| 1 | 00:03.77 | 4 | 86 |
| 2 | 00:04.78 | 5 | 24 |
| 3 | 00:05.78 | 6 | 91 |
| 4 | 00:06.78 | 7 | 57 |
| 5 | 00:07.78 | 8 | 49 |
| 6 | 00:08.77 | 9 | 30 |
| 7 | 00:09.77 | 9 | 30 |
| 8 | 00:10.77 | 11 | 86 |
| 9 | 00:11.77 | 11 | 86 |
| 10 | 00:12.77 | 12 | 59 |
| 11 | 00:13.77 | 13 | 51 |

RALLY TIME bắt đầu ở 2,77 giây khi video mở, tức clip cắt giữa một pha cầu; nhịp đánh khoảng 1 cú mỗi giây. Thẻ năng lượng của người gần camera giảm 94 → 61 và MOVE của người đó tăng 3,8 m → 26,7 m trong cùng khoảng.

### Ngưỡng nghiệm thu (chỉ tính trên clip thật)

| Mô-đun | Đo bằng | Đạt khi |
| --- | --- | --- |
| Sân | Sai số chiếu ngược trên điểm đánh dấu tay (mét) | ≤ 0,25 m trung bình |
| Người chơi | Tỷ lệ frame có đủ 4 người; số lần đổi ID sau khi gán | ≥ 95% và ≤ 2 lần mỗi clip |
| Quãng đường | Sai lệch trên `walk_test` quay thật | ≤ 10% |
| Quả cầu | Tỷ lệ frame đúng vị trí (trong 15 px) trên frame đã gán nhãn | ≥ 70% |
| Cú đánh | Precision và recall, lệch tối đa ±5 frame | ≥ 0,8 |
| Giao diện | 8 mục "giống video mẫu" ở mục 3, ảnh chụp 390×844 | Cả 8 mục đạt khi người duyệt xem ảnh |
| Tốc độ xử lý | Thời gian xử lý chia độ dài clip | ≤ 2× trên GPU, đặt mốc riêng nếu chạy CPU |

## 5. Lộ trình v3

Công sức dồn vào **R1**: đưa ba thứ (video, sân, người) chạy trên ảnh thật trước, vì từ đó HUD và báo cáo mới có dữ liệu đúng để vẽ. Quả cầu (R3) để sau vì là phần rủi ro nhất; HUD (R2) không phụ thuộc vào nó.

Mỗi task là một đơn vị công việc cho agent, cột cuối là điều kiện để được coi là xong. Agent dừng sau mỗi milestone để bạn duyệt.

### R0. Dọn dẹp và đầu vào thật

| Task | Việc cần làm | Đạt khi |
| --- | --- | --- |
| 0.1 | Đổi tên clip giả thành `synthetic_*`; `eval/run.py` in hai bảng `synthetic` và `real`; `STATUS.md` ghi loại dữ liệu | `python -m eval.run --split dev` in hai bảng; không có cổng nào tính trên `synthetic` |
| 0.2 | Xóa `onnx_export.py` và `out/shuttle_head.onnx`; chuyển bộ lọc màu sang `tests/synthetic_helpers.py`; bỏ phát hiện người/cầu bằng màu khỏi pipeline | Chạy pipeline khi thiếu trọng số thì thoát mã ≠ 0 kèm thông báo "Thiếu weights/..., chạy make weights" (có test) |
| 0.3 | `make weights` (và `make.bat` tương đương cho Windows) tải YOLO-Pose và TrackNet, kiểm hash; `make setup` cài `ultralytics`, `torch`, kiểm `ffmpeg` | `make weights` rồi `python -c "from ultralytics import YOLO; YOLO('weights/yolo11n-pose.pt')"` chạy được |
| 0.4 | Sửa `AGENTS.md` theo mục 7; ghi chú "frontend là JS thuần" cho đúng thực tế | `AGENTS.md` không còn nhắc Next.js; có dòng cấm số liệu tự sinh |
| 0.5 | Nhận clip thật, nhãn tay và video mẫu từ bạn (mục 6); trích 12 frame tham chiếu và dựng `fixtures/reference_hud.json` từ bảng ở mục 4 | Qua `schema.validate`; thư mục `fixtures/reference/` có 12 ảnh đặt tên theo giây |

### R1. Sân và người chơi trên ảnh thật

| Task | Việc cần làm | Đạt khi |
| --- | --- | --- |
| 1.1 | Chuẩn hóa bằng ffmpeg: H.264 + AAC, giữ fps, xử lý cờ xoay của iPhone, giới hạn 2 phút / 300 MB; thay `normalize_video_for_browser` | Clip HEVC `.mov` quay dọc phát đúng chiều trong Chromium (Playwright); `ffprobe` báo `h264` |
| 1.2 | Hiệu chỉnh sân linh hoạt: danh sách giao điểm đường kẻ có tên và tọa độ mét (góc, đường giao cầu ngắn/dài, đường giữa, cọc lưới); người dùng chọn ít nhất 4 điểm nhìn thấy; RANSAC khi nhiều hơn 4 | Lỗi chiếu ngược ≤ 0,25 m trên 3 clip thật dev; ảnh lưới chiếu lên frame lưu ở `out/` |
| 1.3 | UI hiệu chỉnh trên cảm ứng: kéo điểm có kính phóng đại, xem trước lưới, hiện lỗi, khóa nút Xử lý khi lỗi quá ngưỡng | Playwright mô phỏng cảm ứng 390×844: chấm 4 điểm thì ra lưới |
| 1.4 | Phát hiện người bằng YOLO-Pose + ByteTrack thật; lọc theo đa giác sân có lề để loại người ở sân bên cạnh và trọng tài (video mẫu có sân phía sau); điểm chân từ mắt cá, dự phòng đáy khung | ≥ 95% frame có đủ 4 người trên clip đôi thật; video gỡ lỗi vẽ khung và chân |
| 1.5 | Gán người chơi trên clip thật: AI đề xuất khung, chạm để nhập tên + đội; nối mảnh track theo vị trí | ≤ 2 lần đổi ID mỗi clip sau khi gán |
| 1.6 | Chỉ số người trên clip thật: `dist_cum_m`, vận tốc, vùng sân; cảnh báo khi máy rung | `walk_test` quay thật sai ≤ 10%; không có vận tốc > 10 m/s; `dist_cum_m` không giảm |

### R2. HUD khớp video mẫu

| Task | Việc cần làm | Đạt khi |
| --- | --- | --- |
| 2.1 | Chạy HUD trên clip thật với dữ liệu người từ R1 (chưa có cầu: LOCK ON và SHOT SPEED hiện `--`) | Trang xem tải được clip thật + `match.json` thật, không cần `MOCK=1` |
| 2.2 | Sửa từng mục HUD theo bảng ở mục 3 (thanh trên, vòng dưới chân, thẻ tên, COURT MAP, bảng người, bố cục dọc); tách `app.js` thành module | Ảnh chụp 390×844 đặt cạnh frame thật của video mẫu (không phải khung giữ chỗ) |
| 2.3 | Đồng bộ khi phát và khi tua | Playwright tua tới 5 thời điểm, `frame_idx` lệch ≤ 1 |
| 2.4 | Dựng lại 12 khung từ `reference_hud.json` trên clip thật để so bố cục và kiểu | Bạn duyệt: 6/8 mục đạt (trừ LOCK ON và SHOT SPEED) |

### R3. Quả cầu, cú đánh, LOCK ON

| Task | Việc cần làm | Đạt khi |
| --- | --- | --- |
| 3.1 | Tích hợp TrackNetV3 thật (cụm 3 frame), xuất `x_px`, `y_px`, `visible`, `conf` theo schema; giữ `clean_shuttle_trajectory` | Báo cáo recall (trong 15 px) trên nhãn tay của clip thật; video gỡ lỗi vẽ cầu |
| 3.2 | Nếu recall < 70%: công cụ gán nhãn nhỏ (bấm vào cầu trên từng frame) rồi tinh chỉnh với 300–500 frame | Recall ≥ 70% trên clip dev, và đo lại trên clip test mà không chỉnh thêm |
| 3.3 | Phát hiện cú đánh trên quỹ đạo thật; phương án dự phòng dùng đỉnh tốc độ cổ tay từ keypoint | Precision và recall ≥ 0,8 (lệch ±5 frame) trên nhãn tay clip thật |
| 3.4 | Tốc độ ước tính: bỏ hệ số 2,1 và việc kẹp; trả `null` khi không chắc; giao diện hiện dấu `~` | Đo tay ít nhất 5 cú (hoặc ghi rõ không có chuẩn) và ghi sai lệch vào báo cáo, không đặt ngưỡng cứng |
| 3.5 | Nối LOCK ON, RALLY COUNT, SHOT SPEED vào HUD | Cả 8 mục HUD đạt khi bạn duyệt ảnh chụp trên clip thật |

### R4. Báo cáo, hiệu năng, xuất video

| Task | Việc cần làm | Đạt khi |
| --- | --- | --- |
| 4.1 | Ghép đầu-cuối trên video thật: tải lên → hiệu chỉnh → xử lý có tiến độ thật → xem trận; lỗi hiện thông báo dễ hiểu | Tải clip thật qua giao diện, không dùng dòng lệnh |
| 4.2 | Trang báo cáo: so sánh người chơi, heatmap, tỷ lệ vùng sân, danh sách pha cầu và cú đánh (bấm để nhảy tới đoạn video), nút tải JSON và CSV | Bấm cú đánh số 5 thì video tới đúng thời điểm ±1 frame; ảnh chụp |
| 4.3 | Hiệu năng: YOLO chạy 15 fps rồi nội suy, TrackNet giữ fps gốc; chỉ làm ONNX/TensorRT thật khi đã đo thấy cần | Thời gian xử lý ≤ 2× độ dài clip trên GPU, ghi rõ máy đo; chỉ số không lệch quá giới hạn so với chạy 30 fps |
| 4.4 | Xuất video có HUD: trình duyệt headless vẽ HUD từng frame, ffmpeg ghép H.264 | mp4 mở được trong Chromium, HUD lệch ≤ 1 frame so với trang xem |
| 4.5 | Nút xóa video và dữ liệu, chạy toàn bộ trên clip test | Bảng nghiệm thu mục 4 trên clip test thật được dán vào `docs/STATUS.md` |

## 6. Việc bạn cần làm

Agent không tự tạo được những thứ dưới đây, và chính việc thiếu chúng đã dẫn tới clip giả. **R0.5 sẽ dừng cho đến khi có ít nhất mục 1 và 2.**

- [ ] **1. Năm clip thật, không có HUD.** 3 clip dev + 2 clip test, mỗi clip 10–20 giây, gốc quay giống video mẫu (đánh đôi, máy đặt sau sân). Nên quay 1080p, máy cố định trên chân; ít nhất một clip là `.mov` HEVC quay dọc từ iPhone để thử chuẩn hóa.
- [ ] **2. File gốc của video mẫu** (`.mov` bạn đã gửi) đặt vào `fixtures/reference/`. Nó chỉ để đối chiếu giao diện.
- [ ] **3. Một clip `walk_test` thật:** một người đi dọc đường biên dài, đúng 13,4 m.
- [ ] **4. Gán nhãn tay** bằng công cụ agent sẽ làm ở R1 và R3: thời điểm cú đánh, 20–30 vị trí quả cầu trên mỗi clip. Không được để agent tự gán nhãn rồi tự chấm.
- [ ] **5. Duyệt ảnh chụp** ở mỗi cổng (R1, R2, R3). Đây là bước quyết định "giống" hay "chưa giống"; con số không thay được mắt người.
