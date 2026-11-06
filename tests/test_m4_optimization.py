import json
import pathlib
import time
import unittest
import cv2
from ai_core.court import auto_detect_court_points, calibrate_court
from ai_core.export_hud_video import export_hud_mp4
from ai_core.pipeline import run_pipeline
from tests.synthetic_helpers import detect_synthetic_players, detect_synthetic_shuttle


class TestM4OptimizationAndExtensions(unittest.TestCase):
    def test_4_1_stride_15fps_speed_and_accuracy_unit(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        vid = root / "fixtures" / "clips" / "synthetic_dev_02.mp4"
        calib = json.loads((root / "fixtures" / "labels" / "synthetic_dev_02.json").read_text(encoding="utf-8"))

        doc_30 = run_pipeline(
            vid,
            calib,
            player_stride=1,
            player_detector_fn=detect_synthetic_players,
            shuttle_detector_fn=detect_synthetic_shuttle,
        )
        t0 = time.perf_counter()
        doc_15 = run_pipeline(
            vid,
            calib,
            player_stride=2,
            player_detector_fn=detect_synthetic_players,
            shuttle_detector_fn=detect_synthetic_shuttle,
        )
        elapsed = time.perf_counter() - t0

        self.assertLessEqual(elapsed, doc_15["video"]["duration_sec"] * 2.0)
        self.assertEqual(len(doc_15["shots"]), len(doc_30["shots"]))
        for pid in ("p1", "p2", "p3", "p4"):
            d30 = doc_30["summary"][pid]["total_distance_m"]
            d15 = doc_15["summary"][pid]["total_distance_m"]
            self.assertLessEqual(abs(d15 - d30) / max(0.1, d30), 0.10)

    def test_4_3_export_mp4_with_hud_zero_frame_lag_unit(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        vid = root / "fixtures" / "clips" / "synthetic_dev_02.mp4"
        calib = json.loads((root / "fixtures" / "labels" / "synthetic_dev_02.json").read_text(encoding="utf-8"))
        match_doc = run_pipeline(
            vid,
            calib,
            player_detector_fn=detect_synthetic_players,
            shuttle_detector_fn=detect_synthetic_shuttle,
        )

        out_mp4 = root / "out" / "hud_export_synthetic_dev_02.mp4"
        info = export_hud_mp4(vid, match_doc, out_mp4)
        self.assertTrue(out_mp4.exists())
        self.assertEqual(info["frame_lag"], 0)

        cap = cv2.VideoCapture(str(out_mp4))
        self.assertTrue(cap.isOpened())
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(count, match_doc["video"]["total_frames"])

    def test_4_4_auto_detect_court_points_unit(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        passed = 0
        for cid in ("synthetic_dev_01", "synthetic_dev_02", "synthetic_dev_03"):
            vid = root / "fixtures" / "clips" / f"{cid}.mp4"
            lbl = json.loads((root / "fixtures" / "labels" / f"{cid}.json").read_text(encoding="utf-8"))
            auto_res = auto_detect_court_points(vid, mode=lbl["mode"])
            if not auto_res["fallback_manual"]:
                cal = calibrate_court(auto_res["calib_points"], mode=lbl["mode"])
                gt_cal = calibrate_court(lbl["calib_points"], mode=lbl["mode"])
                if cal["reproj_error_m"] <= 0.25 and abs(cal["reproj_error_m"] - gt_cal["reproj_error_m"]) <= 0.25:
                    passed += 1
        self.assertGreaterEqual(passed, 2, f"Expected >= 2/3 synthetic clips to pass, got {passed}/3")


if __name__ == "__main__":
    unittest.main()
