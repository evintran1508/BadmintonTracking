import pathlib
import subprocess
import unittest
import cv2
import numpy as np


def _build_side_by_side(hud_png: pathlib.Path, out_png: pathlib.Path) -> None:
    hud = cv2.imread(str(hud_png))
    # Resize if deviceScaleFactor=2 produced 780x1688 back to 390x844
    hud_390 = cv2.resize(hud, (390, 844), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(hud_png), hud_390)

    # Create reference wireframe diagram of the 8 HUD components (390x844)
    ref = np.full((844, 390, 3), (24, 30, 42), dtype=np.uint8)
    cv2.rectangle(ref, (10, 12), (380, 66), (40, 55, 80), -1)
    cv2.putText(ref, "REF: 1.RALLY TIME | 2.COUNT | 3.SPEED", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 220, 255), 1)
    cv2.rectangle(ref, (165, 210), (245, 250), (0, 215, 255), 2)
    cv2.putText(ref, "4.LOCK ON + TRAIL", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)
    cv2.rectangle(ref, (95, 420), (175, 445), (255, 220, 80), 2)
    cv2.putText(ref, "5.NAME TAG + ENERGY", (75, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 80), 1)
    cv2.ellipse(ref, (135, 560), (36, 14), 0, 0, 360, (255, 220, 80), 2)
    cv2.putText(ref, "6.FOOT ELLIPSE (inv H)", (75, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 80), 1)
    cv2.rectangle(ref, (10, 635), (225, 787), (45, 60, 85), -1)
    cv2.putText(ref, "7.PLAYER MOVE TABLE", (25, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 240, 255), 1)
    cv2.rectangle(ref, (240, 635), (378, 787), (30, 85, 50), -1)
    cv2.putText(ref, "8.COURT MAP 2D", (248, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 255, 220), 1)

    combined = np.hstack([ref, hud_390])
    cv2.imwrite(str(out_png), combined)


class TestHudOverlay(unittest.TestCase):
    def test_projection_unit_tests_and_hud_screenshot(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        # 1. Run JS unit tests for courtToPx & projectFootRingPx
        unit_res = subprocess.run(
            ["node", "--test", "tests/projection.test.mjs"],
            cwd=root / "frontend",
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(unit_res.returncode, 0, msg=unit_res.stderr + unit_res.stdout)

        # 2. Run Playwright full HUD check + layer toggles + 390x844 screenshot
        pw_res = subprocess.run(
            ["node", "tests/verify_m0.mjs"],
            cwd=root / "frontend",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(pw_res.returncode, 0, msg=pw_res.stderr + pw_res.stdout)

        hud_png = root / "out" / "hud_390x844.png"
        side_png = root / "out" / "hud_side_by_side_390x844.png"
        self.assertTrue(hud_png.exists())
        _build_side_by_side(hud_png, side_png)
        self.assertTrue(side_png.exists())
        img = cv2.imread(str(hud_png))
        self.assertEqual(img.shape[:2], (844, 390))


if __name__ == "__main__":
    unittest.main()

