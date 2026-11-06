import pathlib
import unittest


class TestRepoStructure(unittest.TestCase):
    def test_required_files_exist(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        for rel in ("AGENTS.md", "Makefile", "docs/PLAN.md", "docs/OPEN_QUESTIONS.md", "config/metrics.yaml"):
            self.assertTrue((root / rel).exists(), f"Missing {rel}")


if __name__ == "__main__":
    unittest.main()

