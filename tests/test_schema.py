import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


VALID_SAMPLE = {
    "schema_version": "2.0",
    "match_id": "match_test",
    "video": {"fps": 30, "total_frames": 2, "duration_sec": 0.067, "width": 886, "height": 1920},
    "court": {
        "mode": "doubles",
        "dims_m": {"width": 6.1, "length": 13.4, "net_y": 6.7},
        "calib_points": [{"px": [100.0, 200.0], "court_m": [0.0, 0.0]}],
        "homography_img_to_court": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "reproj_error_m": 0.12,
    },
    "players": [
        {"id": "p1", "name": "Foam", "team": "A", "color": "#22d3ee"},
    ],
    "rallies": [{"id": 1, "start_frame": 0, "end_frame": 1, "shot_count": 1}],
    "shots": [
        {
            "idx": 1,
            "rally_id": 1,
            "frame": 0,
            "player_id": "p1",
            "speed_kmh": 120.0,
            "speed_confidence": 0.85,
            "from_court_m": [3.0, 10.0],
            "to_court_m": [3.0, 2.0],
        }
    ],
    "frames": [
        {
            "i": 0,
            "t": 0.0,
            "shuttle": {"x_px": 400.0, "y_px": 300.0, "visible": True, "conf": 0.9},
            "players": {
                "p1": {
                    "bbox_px": [300.0, 600.0, 380.0, 820.0],
                    "foot_px": [340.0, 818.0],
                    "foot_court_m": [3.0, 10.0],
                    "speed_mps": 1.0,
                    "dist_cum_m": 1.0,
                    "energy": 95.0,
                    "conf": 0.92,
                }
            },
        },
        {
            "i": 1,
            "t": 0.033,
            "shuttle": {"x_px": 410.0, "y_px": 290.0, "visible": True, "conf": 0.9},
            "players": {
                "p1": {
                    "bbox_px": [302.0, 600.0, 382.0, 820.0],
                    "foot_px": [342.0, 818.0],
                    "foot_court_m": [3.05, 10.0],
                    "speed_mps": 1.5,
                    "dist_cum_m": 1.05,
                    "energy": 94.0,
                    "conf": 0.92,
                }
            },
        },
    ],
    "summary": {
        "p1": {
            "total_distance_m": 1.05,
            "avg_speed_mps": 1.25,
            "max_speed_mps": 1.5,
            "zones_pct": {"front": 20.0, "mid": 60.0, "back": 20.0},
        }
    },
}


class TestSchemaValidation(unittest.TestCase):
    def _run_cli(self, payload: dict) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(payload, f)
            tmp_path = f.name
        try:
            return subprocess.run(
                [sys.executable, "-m", "schema.validate", tmp_path],
                capture_output=True,
                text=True,
                cwd=pathlib.Path(__file__).resolve().parent.parent,
            )
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    def test_valid_payload_passes(self):
        res = self._run_cli(VALID_SAMPLE)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

    def test_decreasing_dist_cum_m_fails(self):
        bad = copy.deepcopy(VALID_SAMPLE)
        bad["frames"][1]["players"]["p1"]["dist_cum_m"] = 0.5
        res = self._run_cli(bad)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("dist_cum_m", res.stderr + res.stdout)

    def test_increasing_energy_fails(self):
        bad = copy.deepcopy(VALID_SAMPLE)
        bad["frames"][1]["players"]["p1"]["energy"] = 99.0
        res = self._run_cli(bad)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("energy", res.stderr + res.stdout)

    def test_generated_artifacts_exist(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        self.assertTrue((root / "schema" / "match.schema.json").exists())
        self.assertTrue((root / "frontend" / "src" / "types" / "match.ts").exists())


if __name__ == "__main__":
    unittest.main()

