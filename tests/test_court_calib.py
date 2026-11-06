import json
import pathlib
import subprocess
import sys
import unittest


class TestCourtCalibration(unittest.TestCase):
    def test_cli_court_calibration_on_dev_clips(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        for clip_id in ("synthetic_dev_01", "synthetic_dev_02", "synthetic_dev_03"):
            video_path = root / "fixtures" / "clips" / f"{clip_id}.mp4"
            calib_path = root / "fixtures" / "labels" / f"{clip_id}.json"
            out_json = out_dir / f"court_{clip_id}.json"
            out_img = out_dir / f"court_grid_{clip_id}.png"

            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ai_core.court",
                    "--video",
                    str(video_path),
                    "--calib",
                    str(calib_path),
                    "--out",
                    str(out_json),
                    "--img",
                    str(out_img),
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_img.exists())

            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertLessEqual(data["reproj_error_m"], 0.25)
            self.assertEqual(len(data["homography_img_to_court"]), 9)

    def test_ransac_rejects_single_outlier(self):
        from ai_core.court import calibrate_court

        root = pathlib.Path(__file__).resolve().parent.parent
        calib = json.loads((root / "fixtures" / "labels" / "synthetic_dev_02.json").read_text(encoding="utf-8"))
        pts = list(calib["calib_points"]) + [{"px": [10.0, 10.0], "court_m": [3.05, 6.7]}]
        result = calibrate_court(pts, mode="doubles")
        self.assertLessEqual(result["reproj_error_m"], 0.25)


if __name__ == "__main__":
    unittest.main()

