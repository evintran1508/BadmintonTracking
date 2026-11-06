import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { invert3x3, courtToPx, projectFootRingPx } from "../src/lib/projection.mjs";
import { createScoreManager } from "../public/audio_score.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const mockPath = path.resolve(__dirname, "../../fixtures/mock_match.json");

test("courtToPx projects court calibration points back to original pixels within 1px", () => {
  const data = JSON.parse(fs.readFileSync(mockPath, "utf-8"));
  const H = data.court.homography_img_to_court;
  for (const pt of data.court.calib_points) {
    const [px, py] = courtToPx(H, pt.court_m[0], pt.court_m[1]);
    assert.ok(Math.abs(px - pt.px[0]) < 1.0, `Expected px=${pt.px[0]}, got ${px}`);
    assert.ok(Math.abs(py - pt.px[1]) < 1.0, `Expected py=${pt.px[1]}, got ${py}`);
  }
});

test("projectFootRingPx produces perspective-foreshortened ellipse (wider than tall)", () => {
  const data = JSON.parse(fs.readFileSync(mockPath, "utf-8"));
  const H = data.court.homography_img_to_court;
  const pts = projectFootRingPx(H, 3.05, 10.0, 0.45, 24);
  assert.equal(pts.length, 24);
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const widthPx = Math.max(...xs) - Math.min(...xs);
  const heightPx = Math.max(...ys) - Math.min(...ys);
  assert.ok(widthPx > heightPx, `Expected perspective foreshortening (width ${widthPx} > height ${heightPx})`);
});

test("invert3x3 throws on singular matrix", () => {
  assert.throws(() => invert3x3([0, 0, 0, 0, 0, 0, 0, 0, 0]), /Singular/);
});

test("createScoreManager handles BWF 21-point rally scoring, undo, and reset", () => {
  const sm = createScoreManager();
  sm.addPoint("A", 30, 1.0);
  sm.addPoint("B", 60, 2.0);
  sm.addPoint("A", 90, 3.0);
  assert.equal(sm.state.teamA, 2);
  assert.equal(sm.state.teamB, 1);
  assert.equal(sm.state.serving, "A");
  sm.undo();
  assert.equal(sm.state.teamA, 1);
  assert.equal(sm.state.teamB, 1);
  assert.equal(sm.state.serving, "B");
  sm.reset();
  assert.equal(sm.state.teamA, 0);
  assert.equal(sm.state.teamB, 0);
  assert.equal(sm.state.history.length, 0);
});

