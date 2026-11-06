import pathlib
import subprocess
import unittest


class TestWebFrameSync(unittest.TestCase):
    def test_video_frame_sync_at_3s(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        res = subprocess.run(
            ["node", "tests/verify_m0.mjs", "--sync-only"],
            cwd=root / "frontend",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(res.returncode, 0, msg=f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        self.assertIn("SYNC_OK: frame_idx=90", res.stdout)


if __name__ == "__main__":
    unittest.main()

