import pathlib
import subprocess
import unittest


class TestCourtCalibUI(unittest.TestCase):
    def test_calib_ui_6_points_grid_and_error_lock(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        res = subprocess.run(
            ["node", "tests/verify_m1_ui.mjs", "--calib-only"],
            cwd=root / "frontend",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
        self.assertIn("CALIB_UI_OK", res.stdout)
        self.assertTrue((root / "out" / "calib_ui_390x844.png").exists())


if __name__ == "__main__":
    unittest.main()

