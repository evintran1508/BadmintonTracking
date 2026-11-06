import { chromium } from "playwright-core";
import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "../..");
const CHROME_PATHS = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
];

function findBrowser() {
  for (const p of CHROME_PATHS) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error("No local Chrome/Edge binary found");
}

async function waitForServer(url, timeoutMs = 8000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error(`Server at ${url} did not start`);
}

async function main() {
  const calibOnly = process.argv.includes("--calib-only");
  const assignOnly = process.argv.includes("--assign-only");
  const port = 3457;
  const server = spawn(process.execPath, ["server.mjs"], {
    cwd: path.resolve(__dirname, ".."),
    env: { ...process.env, MOCK: "1", PORT: String(port) },
    stdio: "ignore",
  });

  try {
    const baseUrl = `http://127.0.0.1:${port}`;
    await waitForServer(`${baseUrl}/api/match`);

    const browser = await chromium.launch({
      executablePath: findBrowser(),
      headless: true,
    });
    const context = await browser.newContext({
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    await page.goto(`${baseUrl}/?tab=setup`, { waitUntil: "networkidle" });

    if (!assignOnly) {
      // 1. Initially 0 points -> button locked
      const initDisabled = await page.isDisabled("#btn-process");
      assert.equal(initDisabled, true, "Process button must be disabled before 6 points are placed");

      // 2. Place 6 valid court points on #calib-canvas
      const validPts = [
        [230, 760],
        [656, 760],
        [171.84, 1014.44],
        [714.16, 1014.44],
        [70, 1460],
        [816, 1460],
      ];
      for (const [px, py] of validPts) {
        await page.evaluate(([x, y]) => window.__addCalibPointPx(x, y), [px, py]);
      }

      const validState = await page.evaluate(() => window.__calibState);
      assert.equal(validState.points.length, 6);
      assert.equal(validState.gridDrawn, true, "Grid preview must be drawn when 6 points are placed");
      assert.ok(validState.reprojErrorM <= 0.25, `Expected error <= 0.25m, got ${validState.reprojErrorM}`);
      assert.equal(await page.isDisabled("#btn-process"), false, "Process button must be enabled when error <= 0.25m");

      const outDir = path.join(ROOT, "out");
      fs.mkdirSync(outDir, { recursive: true });
      await page.screenshot({ path: path.join(outDir, "calib_ui_390x844.png") });

      // 3. Drag point #2 to a bad location -> error > 0.25m -> button locked
      await page.evaluate(() => window.__moveCalibPointPx(2, 480, 1380));
      const badState = await page.evaluate(() => window.__calibState);
      assert.ok(badState.reprojErrorM > 0.25, `Expected error > 0.25m after bad drag, got ${badState.reprojErrorM}`);
      assert.equal(await page.isDisabled("#btn-process"), true, "Process button must be disabled when error > 0.25m");
      console.log(`CALIB_UI_OK: valid_err=${validState.reprojErrorM}m, bad_err=${badState.reprojErrorM}m`);
    }

    if (!calibOnly) {
      // Task 1.4 player assignment UI check
      await page.evaluate(() => {
        window.__resetCalibValid();
        window.__assignPlayerTap(258, 1218, "Foam", "A");
        window.__assignPlayerTap(628, 1244, "Kotty", "A");
        window.__assignPlayerTap(312, 845, "Opp 1", "B");
        window.__assignPlayerTap(574, 830, "Opp 2", "B");
      });
      const assigned = await page.evaluate(() => window.__assignedPlayers);
      assert.equal(assigned.length, 4);
      const outDir = path.join(ROOT, "out");
      fs.mkdirSync(outDir, { recursive: true });
      await page.screenshot({ path: path.join(outDir, "assign_ui_390x844.png") });
      console.log(`ASSIGN_UI_OK: assigned=${assigned.length}`);
    }

    await browser.close();
  } finally {
    server.kill();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
