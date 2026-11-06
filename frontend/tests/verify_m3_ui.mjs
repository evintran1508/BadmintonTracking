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
  const port = 3458;
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
    await page.goto(`${baseUrl}/?tab=summary`, { waitUntil: "networkidle" });

    const outDir = path.join(ROOT, "out");
    fs.mkdirSync(outDir, { recursive: true });
    await page.screenshot({ path: path.join(outDir, "summary_ui_390x844.png") });

    // Click shot #5 and verify video jumps to shot #5 frame within +-1 frame
    const shot5Result = await page.evaluate(async () => {
      const shot5 = window.__matchData.shots.find((s) => s.idx === 5);
      await window.__jumpToShot(5);
      return {
        targetFrame: shot5.frame,
        actualFrame: window.__currentFrameIdx,
        heatmapDrawn: window.__heatmapDrawn,
      };
    });
    assert.equal(shot5Result.heatmapDrawn, true, "2D court heatmap must be rendered");
    assert.ok(
      Math.abs(shot5Result.actualFrame - shot5Result.targetFrame) <= 1,
      `Expected frame ${shot5Result.targetFrame} +-1, got ${shot5Result.actualFrame}`
    );

    // Click Delete Data button
    await page.click("#btn-delete-data");
    const deleted = await page.evaluate(() => window.__dataDeleted);
    assert.equal(deleted, true, "Delete Data button must trigger deletion");

    console.log(`SUMMARY_UI_OK: shot5_frame=${shot5Result.actualFrame}/${shot5Result.targetFrame}`);
    await browser.close();
  } finally {
    server.kill();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

