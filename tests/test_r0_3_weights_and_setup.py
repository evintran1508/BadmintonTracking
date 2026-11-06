import pathlib
import subprocess
import sys
import tempfile
import unittest


class TestR0Task3WeightsAndSetup(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(__file__).resolve().parent.parent

    def test_weights_sha256_verifier_rejects_corrupt_and_passes_real_weights(self):
        from scripts.download_weights import WEIGHT_SPECS, verify_all_weights, verify_sha256

        self.assertIn("yolo11n-pose.pt", WEIGHT_SPECS)
        self.assertIn("tracknetv3.pt", WEIGHT_SPECS)

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"corrupted_weights_bytes")
            corrupt_path = pathlib.Path(tf.name)
        try:
            with self.assertRaises(ValueError):
                verify_sha256(corrupt_path, "0" * 64)
        finally:
            corrupt_path.unlink(missing_ok=True)

        verified = verify_all_weights(self.root / "weights")
        self.assertTrue(verified["yolo11n-pose.pt"])
        self.assertTrue(verified["tracknetv3.pt"])

    def test_yolo11n_pose_loads_via_exact_plan_command(self):
        res = subprocess.run(
            [sys.executable, "-c", "from ultralytics import YOLO; YOLO('weights/yolo11n-pose.pt')"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)

    def test_ffmpeg_and_ffprobe_resolved_with_h264_support(self):
        from ai_core.video_norm import resolve_ffmpeg_binaries

        ffmpeg_bin, ffprobe_bin = resolve_ffmpeg_binaries()
        r1 = subprocess.run([ffmpeg_bin, "-encoders"], capture_output=True, text=True)
        self.assertEqual(r1.returncode, 0)
        self.assertIn("libx264", r1.stdout)

        r2 = subprocess.run([ffprobe_bin, "-version"], capture_output=True, text=True)
        self.assertEqual(r2.returncode, 0)

    def test_tracknetv3_loads_strict_checkpoint(self):
        from ai_core.shuttle import load_tracknet_model

        model, device = load_tracknet_model(self.root / "weights")
        self.assertIsNotNone(model)
        self.assertIn(device, ("cpu", "cuda"))


if __name__ == "__main__":
    unittest.main()

