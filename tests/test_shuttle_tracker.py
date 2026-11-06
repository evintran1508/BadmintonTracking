import json
import math
import pathlib
import unittest
from ai_core.shuttle import clean_shuttle_trajectory, track_shuttle_in_video
from tests.synthetic_helpers import detect_synthetic_shuttle


class TestShuttleTracker(unittest.TestCase):
    def test_unit_shuttle_recall_and_debug_video_on_synthetic(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        dbg_mp4 = out_dir / "debug_shuttle_synthetic_dev_02.mp4"
        lbl = json.loads((root / "fixtures" / "labels" / "synthetic_dev_02.json").read_text(encoding="utf-8"))

        data = track_shuttle_in_video(
            root / "fixtures" / "clips" / "synthetic_dev_02.mp4",
            lbl,
            debug_video_path=dbg_mp4,
            detector_fn=detect_synthetic_shuttle,
        )
        self.assertTrue(dbg_mp4.exists())
        self.assertGreaterEqual(data["recall_15px"], 0.70)

    def test_clean_shuttle_trajectory_outlier_and_gap_interpolation(self):
        raw = []
        for i in range(40):
            x = 200.0 + 4.0 * i
            y = 150.0 + 2.0 * i
            raw.append({"i": i, "x_px": x, "y_px": y, "visible": True, "conf": 0.85})

        raw[10]["x_px"] = 620.0
        raw[10]["y_px"] = 580.0
        for k in range(18, 23):
            raw[k] = {"i": k, "x_px": None, "y_px": None, "visible": False, "conf": 0.0}

        cleaned = clean_shuttle_trajectory(raw, fps=30.0, max_step_px=65.0, max_gap_sec=0.3)
        for k in range(18, 23):
            self.assertTrue(cleaned[k]["visible"], f"Short gap frame {k} should be interpolated")
        for i in range(1, len(cleaned)):
            if cleaned[i]["visible"] and cleaned[i - 1]["visible"]:
                step = math.hypot(
                    cleaned[i]["x_px"] - cleaned[i - 1]["x_px"],
                    cleaned[i]["y_px"] - cleaned[i - 1]["y_px"],
                )
                self.assertLessEqual(step, 65.0)


if __name__ == "__main__":
    unittest.main()
