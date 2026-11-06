import pathlib
import subprocess
import sys
import tempfile
import unittest


class TestR0Task2NoSilentFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(__file__).resolve().parent.parent

    def test_fake_onnx_files_deleted(self):
        self.assertFalse((self.root / "ai_core" / "onnx_export.py").exists())
        self.assertFalse((self.root / "out" / "shuttle_head.onnx").exists())

    def test_color_filters_moved_out_of_ai_core_into_synthetic_helpers(self):
        from tests.synthetic_helpers import detect_synthetic_players, detect_synthetic_shuttle

        self.assertTrue(callable(detect_synthetic_players))
        self.assertTrue(callable(detect_synthetic_shuttle))

        players_src = (self.root / "ai_core" / "players.py").read_text(encoding="utf-8")
        shuttle_src = (self.root / "ai_core" / "shuttle.py").read_text(encoding="utf-8")
        pipeline_src = (self.root / "ai_core" / "pipeline.py").read_text(encoding="utf-8")
        shots_src = (self.root / "ai_core" / "shots.py").read_text(encoding="utf-8")

        self.assertNotIn("_detect_synthetic_silhouettes", players_src)
        self.assertNotIn("_detect_synthetic_silhouettes", pipeline_src)
        self.assertNotIn("yellow_mask", shuttle_src)
        self.assertNotIn("* 2.1", shots_src)
        self.assertNotIn("speed_kmh = 20.0", shots_src)
        self.assertNotIn("speed_kmh = 450.0", shots_src)

    def test_pipeline_exits_nonzero_with_clear_message_when_weights_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_weights = pathlib.Path(tmpdir) / "empty_weights"
            empty_weights.mkdir(parents=True, exist_ok=True)
            out_json = pathlib.Path(tmpdir) / "out.json"
            vid = self.root / "fixtures" / "clips" / "synthetic_dev_02.mp4"
            cal = self.root / "fixtures" / "labels" / "synthetic_dev_02.json"

            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ai_core.pipeline",
                    "--video",
                    str(vid),
                    "--calib",
                    str(cal),
                    "--out",
                    str(out_json),
                    "--weights-dir",
                    str(empty_weights),
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertNotEqual(res.returncode, 0, "Pipeline must fail loudly when weights are missing")
            combined = (res.stdout or "") + (res.stderr or "")
            self.assertIn("Thiếu weights/", combined)
            self.assertIn("chạy make weights", combined)


if __name__ == "__main__":
    unittest.main()

