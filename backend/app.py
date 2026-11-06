from __future__ import annotations
import json
import pathlib
import sqlite3
import threading
import uuid
from typing import Any, Dict, Optional
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ai_core.pipeline import run_pipeline
from ai_core.video_norm import normalize_video_h264

MAX_UPLOAD_MB = 300.0
MAX_DURATION_SEC = 120.0


def normalize_video_for_browser(in_path: pathlib.Path, out_mp4_path: pathlib.Path) -> float:
    return normalize_video_h264(in_path, out_mp4_path, max_duration_sec=MAX_DURATION_SEC)


def create_app(
    storage_dir: Optional[pathlib.Path] = None,
    pipeline_fn: Any = run_pipeline,
) -> FastAPI:
    root = pathlib.Path(__file__).resolve().parent.parent
    base_dir = storage_dir or (root / "out" / "storage")
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "jobs.sqlite"

    def _db() -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress_pct INTEGER NOT NULL DEFAULT 0,
                error_message TEXT DEFAULT '',
                video_path TEXT NOT NULL,
                browser_video_path TEXT NOT NULL,
                calib_json TEXT NOT NULL,
                match_json_path TEXT NOT NULL
            )
            """
        )
        conn.commit()
        return conn

    def _get_job(job_id: str) -> Dict[str, Any]:
        with _db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            return dict(row)

    def _update_job(job_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [job_id]
        with _db() as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", vals)
            conn.commit()

    def _run_worker(job_id: str) -> None:
        try:
            job = _get_job(job_id)
            if job["status"] == "CANCELLED":
                return
            _update_job(job_id, status="PROCESSING", progress_pct=5, error_message="")
            in_vid = pathlib.Path(job["video_path"])
            browser_vid = pathlib.Path(job["browser_video_path"])
            normalize_video_for_browser(in_vid, browser_vid)
            calib_data = json.loads(job["calib_json"])

            def on_prog(pct: int) -> None:
                _update_job(job_id, progress_pct=pct)

            doc = pipeline_fn(in_vid, calib_data, match_id=job_id, progress_cb=on_prog)
            match_out = pathlib.Path(job["match_json_path"])
            match_out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _update_job(job_id, status="COMPLETED", progress_pct=100)
        except Exception as exc:
            _update_job(job_id, status="FAILED", error_message=f"Lỗi xử lý video: {exc}")

    app = FastAPI(title="Badminton AI Backend")

    @app.post("/api/jobs")
    async def create_job(
        video: UploadFile = File(...),
        calib_json: str = Form(...),
        declared_size_mb: float = Form(0.0),
        run_sync: str = Form("0"),
    ) -> Dict[str, Any]:
        raw_bytes = await video.read()
        size_mb = max(declared_size_mb, len(raw_bytes) / (1024 * 1024))
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(status_code=413, detail=f"Dung lượng video ({size_mb:.1f} MB) vượt quá 300 MB")

        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job_dir = base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        in_vid = job_dir / "input.mp4"
        browser_vid = job_dir / "browser.mp4"
        match_out = job_dir / "match.json"
        in_vid.write_bytes(raw_bytes)

        # Quick duration check if readable
        cap = cv2.VideoCapture(str(in_vid))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if frames > 0 and (frames / max(1.0, fps)) > MAX_DURATION_SEC:
                raise HTTPException(status_code=413, detail="Độ dài video vượt quá giới hạn 2 phút (120s)")

        with _db() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, progress_pct, error_message, video_path, browser_video_path, calib_json, match_json_path)
                VALUES (?, 'PENDING', 0, '', ?, ?, ?, ?)
                """,
                (job_id, str(in_vid), str(browser_vid), calib_json, str(match_out)),
            )
            conn.commit()

        if run_sync == "1":
            _run_worker(job_id)
        else:
            threading.Thread(target=_run_worker, args=(job_id,), daemon=True).start()
        return _get_job(job_id)

    @app.get("/api/jobs/{job_id}")
    def get_job_status(job_id: str) -> Dict[str, Any]:
        return _get_job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        _get_job(job_id)
        _update_job(job_id, status="CANCELLED")
        return _get_job(job_id)

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str, run_sync: str = "0") -> Dict[str, Any]:
        _get_job(job_id)
        _update_job(job_id, status="PENDING", progress_pct=0, error_message="")
        if run_sync == "1":
            _run_worker(job_id)
        return _get_job(job_id)

    @app.get("/api/jobs/{job_id}/match")
    def get_job_match(job_id: str) -> Dict[str, Any]:
        job = _get_job(job_id)
        if job["status"] != "COMPLETED":
            raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")
        return json.loads(pathlib.Path(job["match_json_path"]).read_text(encoding="utf-8"))

    @app.get("/api/jobs/{job_id}/video")
    def get_job_video(job_id: str) -> FileResponse:
        job = _get_job(job_id)
        vpath = pathlib.Path(job["browser_video_path"])
        if not vpath.exists():
            vpath = pathlib.Path(job["video_path"])
        media_type = "video/webm" if vpath.suffix.lower() == ".webm" else "video/mp4"
        return FileResponse(str(vpath), media_type=media_type)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> Dict[str, Any]:
        job = _get_job(job_id)
        for key in ("video_path", "browser_video_path", "match_json_path"):
            p = pathlib.Path(job[key])
            p.unlink(missing_ok=True)
            if p.parent.exists() and p.parent != base_dir:
                try:
                    p.parent.rmdir()
                except OSError:
                    pass
        with _db() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
        return {"deleted": job_id}

    return app


app = create_app()

