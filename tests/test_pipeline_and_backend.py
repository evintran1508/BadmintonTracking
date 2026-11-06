import functools
import json
import pathlib
import subprocess
import sys
import unittest
from fastapi.testclient import TestClient
from ai_core.pipeline import run_pipeline
from tests.synthetic_helpers import detect_synthetic_players, detect_synthetic_shuttle


class TestPipelineAndBackend(unittest.TestCase):
    def test_unit_pipeline_outputs_valid_schema_json(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        out_json = root / "out" / "match_synthetic_dev_02.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        calib = json.loads((root / "fixtures" / "labels" / "synthetic_dev_02.json").read_text(encoding="utf-8"))

        doc = run_pipeline(
            root / "fixtures" / "clips" / "synthetic_dev_02.mp4",
            calib,
            player_detector_fn=detect_synthetic_players,
            shuttle_detector_fn=detect_synthetic_shuttle,
        )
        out_json.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        val_res = subprocess.run(
            [sys.executable, "-m", "schema.validate", str(out_json)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(val_res.returncode, 0, msg=val_res.stderr + val_res.stdout)

    def test_backend_upload_413_and_job_lifecycle_and_delete(self):
        from backend.app import create_app

        root = pathlib.Path(__file__).resolve().parent.parent
        syn_pipeline = functools.partial(
            run_pipeline,
            player_detector_fn=detect_synthetic_players,
            shuttle_detector_fn=detect_synthetic_shuttle,
        )
        client = TestClient(create_app(storage_dir=root / "out" / "test_backend", pipeline_fn=syn_pipeline))

        res_413 = client.post(
            "/api/jobs",
            files={"video": ("huge.mp4", b"0000", "video/mp4")},
            data={"calib_json": "{}", "declared_size_mb": "350"},
        )
        self.assertEqual(res_413.status_code, 413)

        bad_res = client.post(
            "/api/jobs",
            files={"video": ("corrupt.mp4", b"not_a_video", "video/mp4")},
            data={"calib_json": json.dumps({"calib_points": []}), "run_sync": "1"},
        )
        self.assertEqual(bad_res.status_code, 200)
        bad_job = bad_res.json()
        self.assertEqual(bad_job["status"], "FAILED")
        self.assertTrue(len(bad_job["error_message"]) > 5)

        retry_res = client.post(f"/api/jobs/{bad_job['job_id']}/retry")
        self.assertEqual(retry_res.json()["status"], "PENDING")
        cancel_res = client.post(f"/api/jobs/{bad_job['job_id']}/cancel")
        self.assertEqual(cancel_res.json()["status"], "CANCELLED")

        vid_bytes = (root / "fixtures" / "clips" / "synthetic_dev_01.mp4").read_bytes()
        calib_str = (root / "fixtures" / "labels" / "synthetic_dev_01.json").read_text(encoding="utf-8")
        ok_res = client.post(
            "/api/jobs",
            files={"video": ("synthetic_dev_01.mp4", vid_bytes, "video/mp4")},
            data={"calib_json": calib_str, "run_sync": "1"},
        )
        self.assertEqual(ok_res.status_code, 200)
        job = ok_res.json()
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["progress_pct"], 100)

        match_res = client.get(f"/api/jobs/{job['job_id']}/match")
        self.assertEqual(match_res.status_code, 200)
        self.assertEqual(match_res.json()["schema_version"], "2.0")

        del_res = client.delete(f"/api/jobs/{job['job_id']}")
        self.assertEqual(del_res.status_code, 200)
        after_del = client.get(f"/api/jobs/{job['job_id']}")
        self.assertEqual(after_del.status_code, 404)


if __name__ == "__main__":
    unittest.main()
