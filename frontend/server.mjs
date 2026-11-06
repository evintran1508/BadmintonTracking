// ponytail: zero-dep Node HTTP server with streaming upload + async Python AI pipeline runner
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.join(__dirname, "public");
const OUT_DIR = path.join(ROOT, "out");
const PORT = Number(process.env.PORT || 3000);
const PYTHON_BIN = process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : (fs.existsSync("/opt/conda/bin/python") ? "/opt/conda/bin/python" : "python3"));

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".webm": "video/webm",
  ".mp4": "video/mp4",
  ".mov": "video/quicktime",
  ".png": "image/png",
};

let activeJob = {
  status: "IDLE",
  progress: 0,
  message: "",
  error: null,
};
let activeChildProcess = null;

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function serveFile(req, res, filePath) {
  if (!fs.existsSync(filePath)) {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
    return;
  }
  const stat = fs.statSync(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME[ext] || "application/octet-stream";

  if (req.headers.range && (ext === ".webm" || ext === ".mp4" || ext === ".mov")) {
    const parts = req.headers.range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
    res.writeHead(206, {
      "Content-Range": `bytes ${start}-${end}/${stat.size}`,
      "Accept-Ranges": "bytes",
      "Content-Length": end - start + 1,
      "Content-Type": contentType,
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
    return;
  }

  const isVideo = ext === ".webm" || ext === ".mp4" || ext === ".mov";
  res.writeHead(200, {
    "Content-Type": contentType,
    "Content-Length": stat.size,
    "Accept-Ranges": "bytes",
    "Cache-Control": isVideo ? "public, max-age=0" : "no-store, no-cache, must-revalidate",
  });
  fs.createReadStream(filePath).pipe(res);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  // 1. Stream video upload directly to disk
  if (url.pathname === "/api/upload-video" && req.method === "POST") {
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const ext = (url.searchParams.get("ext") || ".mp4").replace(/[^a-zA-Z0-9.]/g, "");
    const vidPath = path.join(OUT_DIR, `uploaded_input${ext.startsWith(".") ? ext : ".mp4"}`);
    const latestLink = path.join(OUT_DIR, "uploaded_input.mp4");
    const ws = fs.createWriteStream(vidPath);
    let totalBytes = 0;
    let aborted = false;

    req.on("data", (chunk) => {
      totalBytes += chunk.length;
      if (totalBytes > 300 * 1024 * 1024 && !aborted) {
        aborted = true;
        ws.destroy();
        sendJson(res, 413, { error: "Video vượt quá giới hạn 300 MB" });
      }
    });
    req.pipe(ws);
    ws.on("finish", () => {
      if (aborted) return;
      const extLower = ext.toLowerCase();
      const needsTranscode = extLower === ".mov" || extLower === ".hevc";
      if (!needsTranscode && vidPath !== latestLink) {
        fs.copyFileSync(vidPath, latestLink);
      }
      // Probe video metadata (+ H.264 normalize if .mov) + auto-detect court & YOLO11n-Pose players on frame 0
      const probeScript = `
import json, pathlib, cv2
from ai_core.court import auto_detect_court_points, calibrate_court
from ai_core.players import detect_players_yolo_frame, auto_init_players_from_detections, require_yolo_pose_model, px_to_court_m, select_foot_point, is_inside_court_polygon, select_active_court_players, select_best_seed_frame_dets
from ai_core.video_norm import normalize_video_h264
import numpy as np
src = pathlib.Path(r"${vidPath.replace(/\\/g, "\\\\")}")
p = pathlib.Path(r"${latestLink.replace(/\\/g, "\\\\")}")
if ${needsTranscode ? "True" : "False"}:
    normalize_video_h264(src, p, max_duration_sec=600.0)
cap = cv2.VideoCapture(str(p))
if not cap.isOpened():
    normalize_video_h264(src, p, max_duration_sec=600.0)
    cap = cv2.VideoCapture(str(p))
fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
ok, frame0 = cap.read()
if not ok or vw <= 0 or vh <= 0:
    cap.release()
    raise ValueError("Video tải lên không đọc được khung hình hợp lệ")
auto_c = auto_detect_court_points(p, mode="doubles")
cal = calibrate_court(auto_c["calib_points"], mode="doubles")
H = np.array(cal["homography_img_to_court"], dtype=np.float64).reshape(3, 3)
ym = require_yolo_pose_model()
sample_frames = []
for f_idx in (0, 15, 30, 50):
    if f_idx > 0 and n > f_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok_f, fr = cap.read()
    else:
        ok_f, fr = (True, frame0) if f_idx == 0 else (False, None)
    if not ok_f or fr is None:
        continue
    dets = detect_players_yolo_frame(fr, ym)
    filt = []
    for d in dets:
        fx, fy, _ = select_foot_point(d["bbox_px"], d.get("keypoints"))
        cm = px_to_court_m(H, fx, fy)
        if is_inside_court_polygon(cm, cal["dims_m"], margin_m=0.85):
            filt.append({"foot_px": [fx, fy], "foot_court_m": [cm[0], cm[1]], "bbox_px": d["bbox_px"], "conf": d["conf"]})
    sample_frames.append(select_active_court_players(filt, cal["dims_m"], mode="doubles"))
cap.release()
best_seed = select_best_seed_frame_dets(sample_frames, mode="doubles", dims_m=cal["dims_m"])
players_init = auto_init_players_from_detections(best_seed, mode="doubles", dims_m=cal["dims_m"]) if best_seed else []
print(json.dumps({"width": vw, "height": vh, "fps": round(fps, 2), "frames": n, "durationSec": round(n/max(1.0,fps), 2), "transcoded": ${needsTranscode ? "True" : "False"}, "calib_points": auto_c["calib_points"], "fallback_manual": auto_c["fallback_manual"], "initial_players": players_init}))
`;
      const probe = spawnSync(PYTHON_BIN, ["-c", probeScript], { cwd: ROOT, encoding: "utf-8" });
      if (probe.status !== 0 || !probe.stdout) {
        const errTail = (probe.stderr || "Lỗi kiểm tra video tải lên").trim().slice(-320);
        return sendJson(res, 422, { ok: false, error: errTail });
      }
      const lines = probe.stdout.trim().split(/\r?\n/);
      const meta = JSON.parse(lines[lines.length - 1]);
      sendJson(res, 200, {
        ok: true,
        videoUrl: `/uploaded_input.mp4?v=${Date.now()}`,
        sizeBytes: totalBytes,
        ...meta,
      });
    });
    ws.on("error", (err) => {
      if (!aborted) sendJson(res, 500, { error: String(err) });
    });
    return;
  }

  // 1b. On-demand Auto-Calibrate Court & Auto-Detect Players on active video
  if (url.pathname === "/api/auto-setup" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      try {
        const payload = body ? JSON.parse(body) : {};
        const mode = payload.mode === "singles" ? "singles" : "doubles";
        const uploadedVid = path.join(OUT_DIR, "uploaded_input.mp4");
        const refVid = fs.existsSync(path.join(ROOT, "fixtures", "reference", "reference_match.mp4"))
          ? path.join(ROOT, "fixtures", "reference", "reference_match.mp4")
          : path.join(ROOT, "fixtures", "clips", "synthetic_dev_02.mp4");
        const videoToUse = payload.useUploadedVideo && fs.existsSync(uploadedVid) ? uploadedVid : (fs.existsSync(uploadedVid) ? uploadedVid : refVid);
        const autoScript = `
import json, pathlib, cv2, numpy as np
from ai_core.court import auto_detect_court_points, calibrate_court
from ai_core.players import detect_players_yolo_frame, auto_init_players_from_detections, require_yolo_pose_model, px_to_court_m, select_foot_point, is_inside_court_polygon, select_active_court_players, select_best_seed_frame_dets
p = pathlib.Path(r"${videoToUse.replace(/\\/g, "\\\\")}")
auto_c = auto_detect_court_points(p, mode="${mode}")
cal = calibrate_court(auto_c["calib_points"], mode="${mode}")
H = np.array(cal["homography_img_to_court"], dtype=np.float64).reshape(3, 3)
ym = require_yolo_pose_model()
cap = cv2.VideoCapture(str(p))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
sample_frames = []
for f_idx in (0, 15, 30, 50):
    if f_idx > 0 and n > f_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ok_f, fr = cap.read()
    if not ok_f or fr is None:
        continue
    dets = detect_players_yolo_frame(fr, ym)
    filt = []
    for d in dets:
        fx, fy, _ = select_foot_point(d["bbox_px"], d.get("keypoints"))
        cm = px_to_court_m(H, fx, fy)
        if is_inside_court_polygon(cm, cal["dims_m"], margin_m=0.85):
            filt.append({"foot_px": [fx, fy], "foot_court_m": [cm[0], cm[1]], "bbox_px": d["bbox_px"], "conf": d["conf"]})
    sample_frames.append(select_active_court_players(filt, cal["dims_m"], mode="${mode}"))
cap.release()
best_seed = select_best_seed_frame_dets(sample_frames, mode="${mode}", dims_m=cal["dims_m"])
players_init = auto_init_players_from_detections(best_seed, mode="${mode}", dims_m=cal["dims_m"]) if best_seed else []
print(json.dumps({"ok": True, "calib_points": auto_c["calib_points"], "fallback_manual": auto_c["fallback_manual"], "initial_players": players_init}))
`;
        const probe = spawnSync(PYTHON_BIN, ["-c", autoScript], { cwd: ROOT, encoding: "utf-8" });
        if (probe.status !== 0 || !probe.stdout) {
          const errTail = (probe.stderr || "Lỗi tự động canh sân & người").trim().slice(-320);
          return sendJson(res, 422, { ok: false, error: errTail });
        }
        const lines = probe.stdout.trim().split(/\r?\n/);
        const meta = JSON.parse(lines[lines.length - 1]);
        sendJson(res, 200, meta);
      } catch (err) {
        sendJson(res, 500, { ok: false, error: String(err) });
      }
    });
    return;
  }

  // 2. Start AI pipeline asynchronously (with live progress polling)
  if (url.pathname === "/api/cancel-process" && req.method === "POST") {
    if (activeChildProcess && !activeChildProcess.killed) {
      activeJob.status = "CANCELLED";
      activeJob.message = "Đã hủy xử lý AI.";
      try {
        if (process.platform === "win32") {
          spawnSync("taskkill", ["/PID", String(activeChildProcess.pid), "/T", "/F"]);
        } else {
          activeChildProcess.kill("SIGTERM");
        }
      } catch (err) {}
      activeChildProcess = null;
    }
    return sendJson(res, 200, { ok: true, status: "CANCELLED" });
  }

  if (url.pathname === "/api/process" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        if (activeChildProcess && !activeChildProcess.killed) {
          try {
            if (process.platform === "win32") {
              spawnSync("taskkill", ["/PID", String(activeChildProcess.pid), "/T", "/F"]);
            } else {
              activeChildProcess.kill("SIGTERM");
            }
          } catch (err) {}
          activeChildProcess = null;
        }

        fs.mkdirSync(OUT_DIR, { recursive: true });
        const calibData = JSON.parse(body || "{}");
        const calibPath = path.join(OUT_DIR, "uploaded_calib.json");
        const matchOutPath = path.join(OUT_DIR, "uploaded_match.json");
        const uploadedVid = path.join(OUT_DIR, "uploaded_input.mp4");
        const fallbackVid = path.join(ROOT, "fixtures", "clips", "synthetic_dev_02.mp4");
        const targetVid = fs.existsSync(uploadedVid) ? uploadedVid : fallbackVid;
        const maxFrames = calibData.max_frames !== undefined ? Number(calibData.max_frames) : 450;

        fs.writeFileSync(calibPath, JSON.stringify(calibData, null, 2), "utf-8");
        activeJob = {
          status: "PROCESSING",
          progress: 8,
          message: "Đang nạp mô hình & hiệu chỉnh tọa độ sân...",
          error: null,
        };

        const child = spawn(
          PYTHON_BIN,
          [
            "-u",
            "-m",
            "ai_core.pipeline",
            "--video",
            targetVid,
            "--calib",
            calibPath,
            "--out",
            matchOutPath,
            "--player-stride",
            "2",
            "--max-frames",
            String(maxFrames),
          ],
          { cwd: ROOT }
        );
        activeChildProcess = child;

        let stderrBuf = "";
        let stdoutBuf = "";
        child.stdout.on("data", (d) => {
          const str = d.toString();
          process.stdout.write(str);
          stdoutBuf += str;
          const matches = [...stdoutBuf.matchAll(/PROGRESS:(\d+)/g)];
          if (matches.length > 0) {
            const pct = Number(matches[matches.length - 1][1]);
            if (activeJob.status === "PROCESSING" && pct >= activeJob.progress) {
              activeJob.progress = Math.min(99, pct);
            }
          }
          const statusMatches = [...stdoutBuf.matchAll(/STATUS:([^\r\n]+)/g)];
          if (statusMatches.length > 0) {
            activeJob.message = statusMatches[statusMatches.length - 1][1].trim();
          }
        });
        child.stderr.on("data", (d) => {
          const str = d.toString();
          process.stderr.write(str);
          stderrBuf += str;
        });
        child.on("close", (code) => {
          activeChildProcess = null;
          if (activeJob.status === "CANCELLED") {
            return;
          }
          if (code === 0 && fs.existsSync(matchOutPath)) {
            activeJob = { status: "COMPLETED", progress: 100, message: "Hoàn tất phân tích AI!", error: null };
          } else {
            activeJob = {
              status: "FAILED",
              progress: 0,
              message: "Lỗi phân tích AI",
              error: stderrBuf.slice(-400) || `Exit code ${code}`,
            };
          }
        });

        sendJson(res, 200, { ok: true, status: "PROCESSING" });
      } catch (err) {
        sendJson(res, 500, { error: String(err) });
      }
    });
    return;
  }

  if (url.pathname === "/api/process-status" && req.method === "GET") {
    const matchOutPath = path.join(OUT_DIR, "uploaded_match.json");
    if (activeJob.status === "COMPLETED" && fs.existsSync(matchOutPath)) {
      const matchJson = JSON.parse(fs.readFileSync(matchOutPath, "utf-8"));
      return sendJson(res, 200, { ...activeJob, match: matchJson });
    }
    return sendJson(res, 200, activeJob);
  }

  if (url.pathname === "/uploaded_input.mp4") {
    return serveFile(req, res, path.join(OUT_DIR, "uploaded_input.mp4"));
  }

  if (url.pathname === "/api/match") {
    if (req.method === "DELETE") {
      return sendJson(res, 200, { deleted: true });
    }
    const uploadedMatch = path.join(OUT_DIR, "uploaded_match.json");
    const realOut = path.join(OUT_DIR, "match_real_dev_01.json");
    const srcParam = url.searchParams.get("source");
    if ((srcParam === "uploaded" || srcParam === "auto") && fs.existsSync(uploadedMatch)) {
      return serveFile(req, res, uploadedMatch);
    }
    const useMock = (process.env.MOCK || "1") === "1" || !fs.existsSync(realOut);
    const targetFile = useMock ? path.join(ROOT, "fixtures", "mock_match.json") : realOut;
    return serveFile(req, res, targetFile);
  }

  if (url.pathname.startsWith("/api/")) {
    return sendJson(res, 404, { error: `API route not found: ${url.pathname}` });
  }

  if (url.pathname === "/mock_video.webm") {
    return serveFile(req, res, path.join(ROOT, "fixtures", "mock_video.webm"));
  }
  if (url.pathname === "/src/lib/projection.mjs") {
    return serveFile(req, res, path.join(__dirname, "src", "lib", "projection.mjs"));
  }
  const rel = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
  return serveFile(req, res, path.join(PUBLIC_DIR, rel));
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Badminton AI Studio running on http://127.0.0.1:${PORT}`);
});
