import pathlib
import subprocess
import unittest


class TestSummaryUI(unittest.TestCase):
    def test_summary_heatmap_and_shot_jump_within_1_frame(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        res = subprocess.run(
            ["node", "tests/verify_m3_ui.mjs"],
            cwd=root / "frontend",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
        self.assertIn("SUMMARY_UI_OK", res.stdout)
        self.assertTrue((root / "out" / "summary_ui_390x844.png").exists())


if __name__ == "__main__":
    unittest.main()

