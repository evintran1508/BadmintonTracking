import json
import pathlib
import unittest
from ai_core.players import select_foot_point, track_players_in_video
from tests.synthetic_helpers import detect_synthetic_players


class TestPlayersTracker(unittest.TestCase):
    def test_select_foot_point_ankle_and_bbox_fallback(self):
        bbox = [100.0, 200.0, 160.0, 340.0]
        kps_good = [[0.0, 0.0, 0.0]] * 15 + [[122.0, 334.0, 0.9], [138.0, 336.0, 0.8]]
        fx, fy, used_ankle = select_foot_point(bbox, kps_good)
        self.assertTrue(used_ankle)
        self.assertAlmostEqual(fx, 130.0)
        self.assertAlmostEqual(fy, 335.0)

        kps_occ = [[0.0, 0.0, 0.0]] * 15 + [[122.0, 334.0, 0.2], [138.0, 336.0, 0.1]]
        fx2, fy2, used_ankle2 = select_foot_point(bbox, kps_occ)
        self.assertFalse(used_ankle2)
        self.assertAlmostEqual(fx2, 130.0)
        self.assertAlmostEqual(fy2, 340.0)

    def test_unit_players_on_synthetic_doubles_clips(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        for clip_id in ("synthetic_dev_02", "synthetic_dev_03"):
            video_path = root / "fixtures" / "clips" / f"{clip_id}.mp4"
            calib = json.loads((root / "fixtures" / "labels" / f"{clip_id}.json").read_text(encoding="utf-8"))
            debug_mp4 = out_dir / f"debug_players_{clip_id}.mp4"
            out_json = out_dir / f"players_{clip_id}.json"

            data = track_players_in_video(
                video_path,
                calib,
                debug_video_path=debug_mp4,
                detector_fn=detect_synthetic_players,
            )
            out_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            self.assertTrue(debug_mp4.exists())
            self.assertGreater(debug_mp4.stat().st_size, 1000)
            self.assertGreaterEqual(data["coverage_4_players_ratio"], 0.95)
            for f in data["frames"]:
                self.assertLessEqual(len(f["detections"]), 4)


if __name__ == "__main__":
    unittest.main()
