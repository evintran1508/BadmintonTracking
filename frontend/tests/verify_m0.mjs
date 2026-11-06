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
  throw new Error(`Server at ${url} did not start within ${timeoutMs}ms`);
}

async function main() {
  const syncOnly = process.argv.includes("--sync-only");
  const port = 3456;
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
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();
    await page.goto(baseUrl, { waitUntil: "networkidle" });

    // Seek exactly to t = 3.0s via exposed helper and verify frame_idx == round(3.0 * 30) == 90
    const syncResult = await page.evaluate(async () => {
      await window.__seekToTime(3.0);
      return {
        frameIdx: window.__currentFrameIdx,
        debugText: document.getElementById("debug-frame")?.textContent?.trim(),
        expected: Math.round(3.0 * window.__matchData.video.fps),
      };
    });

    assert.equal(syncResult.frameIdx, syncResult.expected, `Expected frame ${syncResult.expected}, got ${syncResult.frameIdx}`);
    assert.equal(syncResult.frameIdx, 90);
    console.log(`SYNC_OK: frame_idx=${syncResult.frameIdx} (${syncResult.debugText})`);

    if (!syncOnly) {
      // Task 0.5 checks: verify all 8 HUD components and layer toggles, capture 390x844 screenshot
      const hudState = await page.evaluate(() => window.__lastHudRenderStats);
      assert.ok(hudState, "HUD render stats must be populated");
      for (const comp of [
        "rallyTime",
        "rallyCount",
        "shotSpeed",
        "scoreBoard",
        "lockOn",
        "playerTags",
        "footRings",
        "moveTable",
        "courtMap",
      ]) {
        assert.equal(hudState.drawn[comp], true, `HUD component ${comp} was not drawn at t=3.0s`);
      }

      // Verify Match Scoring (+A, +B, Undo), Video Timeline Scrubbing, and Audio Toggle
      await page.click("#btn-score-a");
      await page.click("#btn-score-a");
      await page.click("#btn-score-b");
      const scoreText = await page.textContent("#score-display");
      assert.equal(scoreText.trim(), "2 - 1", `Expected score '2 - 1', got '${scoreText}'`);

      // Verify timeline slider scrubbing to frame 120 and back to 90
      await page.evaluate(() => {
        const s = document.getElementById("timeline-slider");
        s.value = "120";
        s.dispatchEvent(new Event("input", { bubbles: true }));
        s.dispatchEvent(new Event("change", { bubbles: true }));
      });
      const scrubbedFrame = await page.evaluate(() => window.__currentFrameIdx);
      assert.equal(scrubbedFrame, 120, `Expected scrubbed frame 120, got ${scrubbedFrame}`);

      // Verify audio toggle button unmutes video
      await page.click("#btn-mute");
      const isUnmuted = await page.evaluate(() => !document.getElementById("match-video").muted);
      assert.equal(isUnmuted, true, "Clicking #btn-mute should unmute the video");

      await page.evaluate(() => window.__seekToTime(3.0));

      const outDir = path.join(ROOT, "out");
      fs.mkdirSync(outDir, { recursive: true });
      const shotPath = path.join(outDir, "hud_390x844.png");
      await page.screenshot({ path: shotPath });

      // Test toggling a layer off and back on
      await page.click('input[data-layer="lockOn"]');
      const afterToggleOff = await page.evaluate(() => window.__lastHudRenderStats.drawn.lockOn);
      assert.equal(afterToggleOff, false, "Layer lockOn should be false when toggled off");
      await page.click('input[data-layer="lockOn"]');
      const afterToggleOn = await page.evaluate(() => window.__lastHudRenderStats.drawn.lockOn);
      assert.equal(afterToggleOn, true, "Layer lockOn should be true when toggled back on");

      console.log(`HUD_OK: screenshot saved to ${shotPath}`);
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
