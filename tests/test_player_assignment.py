import json
import pathlib
import subprocess
import unittest
import numpy as np
from ai_core.players import assign_and_stitch_tracks, track_players_in_video
from tests.synthetic_helpers import detect_synthetic_players


class TestPlayerAssignmentAndStitching(unittest.TestCase):
    def test_ui_player_assignment_playwright(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        res = subprocess.run(
            ["node", "tests/verify_m1_ui.mjs", "--assign-only"],
            cwd=root / "frontend",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
        self.assertIn("ASSIGN_UI_OK: assigned=4", res.stdout)
        self.assertTrue((root / "out" / "assign_ui_390x844.png").exists())

    def test_stitch_fragmented_tracks_under_2_switches(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        for clip_id in ("synthetic_dev_02", "synthetic_dev_03"):
            video_path = root / "fixtures" / "clips" / f"{clip_id}.mp4"
            calib = json.loads((root / "fixtures" / "labels" / f"{clip_id}.json").read_text(encoding="utf-8"))
            data = track_players_in_video(video_path, calib, detector_fn=detect_synthetic_players)
            self.assertLessEqual(data["id_switches"], 2)

        H = np.eye(3, dtype=np.float64)
        init_players = [
            {"id": "p1", "name": "Foam", "team": "A", "click_px": [1.5, 10.0]},
            {"id": "p2", "name": "Kotty", "team": "A", "click_px": [4.5, 10.0]},
        ]
        frames_dets = []
        for f in range(20):
            tid_offset = 100 if f >= 10 else 0
            frames_dets.append(
                [
                    {"raw_track_id": 1 + tid_offset, "foot_court_m": [1.5 + 0.02 * f, 10.0]},
                    {"raw_track_id": 2 + tid_offset, "foot_court_m": [4.5 - 0.02 * f, 10.0]},
                ]
            )
        assigned, switches = assign_and_stitch_tracks(frames_dets, init_players, H)
        self.assertEqual(switches, 0)
        self.assertEqual(len(assigned), 20)
        self.assertAlmostEqual(assigned[19]["p1"]["foot_court_m"][0], 1.5 + 0.02 * 19)
        self.assertAlmostEqual(assigned[19]["p2"]["foot_court_m"][0], 4.5 - 0.02 * 19)


if __name__ == "__main__":
    unittest.main()
