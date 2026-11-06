import json
import pathlib
import subprocess
import sys
import unittest


class TestR0Task5ReferenceHud(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(__file__).resolve().parent.parent

    def test_twelve_reference_frames_extracted_by_second(self):
        ref_dir = self.root / "fixtures" / "reference"
        for sec in range(12):
            img_path = ref_dir / f"sec_{sec:02d}.jpg"
            self.assertTrue(img_path.exists(), f"Missing reference image {img_path.name}")
            self.assertGreater(img_path.stat().st_size, 5000)

    def test_reference_hud_json_passes_schema_validate_and_matches_plan_table(self):
        ref_json = self.root / "fixtures" / "reference_hud.json"
        self.assertTrue(ref_json.exists())

        res = subprocess.run(
            [sys.executable, "-m", "schema.validate", str(ref_json)],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)

        doc = json.loads(ref_json.read_text(encoding="utf-8"))
        self.assertEqual(doc["rallies"][0]["shot_count"], 13)
        obs_speeds = [s["speed_kmh"] for s in doc["shots"] if s["speed_kmh"] is not None]
        for expected_spd in (214.0, 86.0, 24.0, 91.0, 57.0, 49.0, 30.0, 59.0, 51.0):
            self.assertIn(expected_spd, obs_speeds)


if __name__ == "__main__":
    unittest.main()

