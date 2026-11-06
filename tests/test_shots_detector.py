import json
import pathlib
import unittest
import numpy as np
from ai_core.court import calibrate_court
from ai_core.players import track_players_in_video
from ai_core.shots import detect_shots_and_rallies, evaluate_shots
from ai_core.shuttle import track_shuttle_in_video
from tests.synthetic_helpers import detect_synthetic_players, detect_synthetic_shuttle


class TestShotsDetector(unittest.TestCase):
    def test_unit_shots_precision_recall_and_no_clamping(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        vid = root / "fixtures" / "clips" / "synthetic_dev_02.mp4"
        calib = json.loads((root / "fixtures" / "labels" / "synthetic_dev_02.json").read_text(encoding="utf-8"))
        court_meta = calibrate_court(calib["calib_points"], mode=calib["mode"])
        H = np.array(court_meta["homography_img_to_court"], dtype=np.float64).reshape(3, 3)

        sh_res = track_shuttle_in_video(vid, calib, detector_fn=detect_synthetic_shuttle)
        pl_res = track_players_in_video(vid, calib, detector_fn=detect_synthetic_players)
        shots, rallies = detect_shots_and_rallies(sh_res["frames"], pl_res["frames"], H, fps=sh_res["fps"])
        metrics = evaluate_shots(shots, calib.get("gt_shots", []))

        self.assertGreaterEqual(metrics["precision"], 0.80)
        self.assertGreaterEqual(metrics["recall"], 0.80)
        self.assertGreaterEqual(len(rallies), 1)

        for s in shots:
            if s["speed_kmh"] is not None:
                self.assertGreaterEqual(s["speed_kmh"], 10.0)
                self.assertLessEqual(s["speed_kmh"], 450.0)
            self.assertGreaterEqual(s["speed_confidence"], 0.0)
            self.assertLessEqual(s["speed_confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
