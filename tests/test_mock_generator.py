import hashlib
import json
import pathlib
import subprocess
import sys
import unittest


class TestMockMatchGenerator(unittest.TestCase):
    def test_deterministic_hash_and_validation(self):
        from fixtures.generate_mock import generate_mock_dict

        d1 = json.dumps(generate_mock_dict(seed=42), sort_keys=True)
        d2 = json.dumps(generate_mock_dict(seed=42), sort_keys=True)
        h1 = hashlib.sha256(d1.encode("utf-8")).hexdigest()
        h2 = hashlib.sha256(d2.encode("utf-8")).hexdigest()
        self.assertEqual(h1, h2)

        root = pathlib.Path(__file__).resolve().parent.parent
        mock_path = root / "fixtures" / "mock_match.json"
        self.assertTrue(mock_path.exists(), "fixtures/mock_match.json must exist")

        data = json.loads(mock_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["players"]), 4)
        self.assertEqual(data["video"]["duration_sec"], 15.0)
        self.assertEqual(data["video"]["total_frames"], 450)
        self.assertEqual(len(data["shots"]), 12)

        res = subprocess.run(
            [sys.executable, "-m", "schema.validate", str(mock_path)],
            capture_output=True,
            text=True,
            cwd=root,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)


if __name__ == "__main__":
    unittest.main()

