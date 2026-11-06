import json
import pathlib
import unittest
from ai_core.metrics import compute_player_metrics
from ai_core.players import track_players_in_video
from tests.synthetic_helpers import detect_synthetic_players


class TestPlayerMetrics(unittest.TestCase):
    def test_synthetic_walk_test_distance_within_10_percent(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        calib = json.loads((root / "fixtures" / "labels" / "synthetic_walk_test.json").read_text(encoding="utf-8"))
        tracked = track_players_in_video(
            root / "fixtures" / "synthetic_walk_test.mp4",
            calib,
            detector_fn=detect_synthetic_players,
        )
        raw_trajs = {
            "p1": [
                tuple(f["detections"][0]["foot_court_m"]) if f["detections"] else None
                for f in tracked["frames"]
            ]
        }
        frames_out, summary = compute_player_metrics(raw_trajs, fps=30.0)
        dist_m = summary["p1"]["total_distance_m"]
        rel_err = abs(dist_m - 13.4) / 13.4
        self.assertLessEqual(rel_err, 0.10, f"synthetic_walk_test distance {dist_m}m off by {rel_err*100:.1f}%")

        prev_d = 0.0
        prev_e = 100.0
        for f in frames_out:
            pf = f["p1"]
            self.assertLessEqual(pf["speed_mps"], 10.0)
            self.assertGreaterEqual(pf["dist_cum_m"], prev_d)
            self.assertLessEqual(pf["energy"], prev_e)
            prev_d = pf["dist_cum_m"]
            prev_e = pf["energy"]

    def test_savgol_smoothing_suppresses_jitter_and_caps_velocity(self):
        raw_pts = [(1.5, 10.0 + 0.05 * i) for i in range(30)]
        raw_pts[15] = (4.8, 12.5)
        frames_out, summary = compute_player_metrics({"p1": raw_pts}, fps=30.0)
        for f in frames_out:
            self.assertLessEqual(f["p1"]["speed_mps"], 10.0)
        z = summary["p1"]["zones_pct"]
        self.assertAlmostEqual(z["front"] + z["mid"] + z["back"], 100.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
