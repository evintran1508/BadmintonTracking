import pathlib
import subprocess
import sys
import unittest


class TestR0Task1SyntheticAndEval(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(__file__).resolve().parent.parent

    def test_agents_md_updated_for_v3(self):
        agents_txt = (self.root / "AGENTS.md").read_text(encoding="utf-8")
        self.assertNotIn("Next.js", agents_txt)
        self.assertIn("CẤM tự sinh clip/nhãn/số liệu", agents_txt)
        self.assertIn("JS thuần", agents_txt)

    def test_synthetic_clips_renamed_and_no_unprefixed_clips_remain(self):
        clips_dir = self.root / "fixtures" / "clips"
        labels_dir = self.root / "fixtures" / "labels"

        expected_clips = (
            "synthetic_dev_01.mp4",
            "synthetic_dev_02.mp4",
            "synthetic_dev_03.mp4",
            "synthetic_test_01.mp4",
            "synthetic_test_02.mp4",
        )
        for cname in expected_clips:
            self.assertTrue((clips_dir / cname).exists(), f"Missing {cname}")

        self.assertTrue((self.root / "fixtures" / "synthetic_walk_test.mp4").exists())
        self.assertTrue((labels_dir / "synthetic_walk_test.json").exists())

        forbidden_old = ("dev_01.mp4", "dev_02.mp4", "dev_03.mp4", "test_01.mp4", "test_02.mp4")
        for old_name in forbidden_old:
            self.assertFalse((clips_dir / old_name).exists(), f"Old unprefixed clip still exists: {old_name}")
        self.assertFalse((self.root / "fixtures" / "walk_test.mp4").exists())

    def test_eval_prints_two_tables_and_never_counts_synthetic_as_gate_pass(self):
        res = subprocess.run(
            [sys.executable, "-m", "eval.run", "--split", "dev"],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out = (res.stdout or "") + (res.stderr or "")
        self.assertIn("synthetic", out.lower())
        self.assertIn("real", out.lower())
        # Count markdown table headers "| Mô-đun |" -> must print 2 tables (synthetic + real)
        self.assertGreaterEqual(out.count("| Mô-đun |"), 2, f"Expected 2 tables in eval output:\n{out}")

        # If real_dev_* clips are absent, eval gate MUST NOT pass on synthetic alone
        real_dev_clips = list((self.root / "fixtures" / "clips").glob("real_dev_*.mp4"))
        if not real_dev_clips:
            self.assertNotEqual(res.returncode, 0, "eval.run must not return 0 when real clips are missing")

    def test_status_md_labels_data_type(self):
        status_txt = (self.root / "docs" / "STATUS.md").read_text(encoding="utf-8")
        self.assertTrue("clip giả" in status_txt or "clip thật" in status_txt)


if __name__ == "__main__":
    unittest.main()

