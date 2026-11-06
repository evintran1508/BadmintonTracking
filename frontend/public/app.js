import { courtToPx, projectFootRingPx, solveCourtCalibration } from "/src/lib/projection.mjs";
import { createScoreManager, playScoreChimeSfx, playShuttleHitSfx } from "/audio_score.mjs";

const stageWrap = document.getElementById("stage-wrap");
const stageEl = document.getElementById("stage");
const video = document.getElementById("match-video");
const canvas = document.getElementById("hud-canvas");
const ctx = canvas.getContext("2d");
const calibCanvas = document.getElementById("calib-canvas");
const calibCtx = calibCanvas.getContext("2d");
const calibBanner = document.getElementById("calib-banner");

const studioDock = document.getElementById("studio-dock");
const setupPanel = document.getElementById("setup-panel");
const summaryPanel = document.getElementById("summary-panel");
const tabBtnSetup = document.getElementById("tab-btn-setup");
const tabBtnSummary = document.getElementById("tab-btn-summary");
const modePill = document.getElementById("mode-pill");

const calibStatusEl = document.getElementById("calib-status");
const assignStatusEl = document.getElementById("assign-status");
const playerEditListEl = document.getElementById("player-edit-list");
const btnProcess = document.getElementById("btn-process");
const btnCancelProcess = document.getElementById("btn-cancel-process");
const btnToggleCalib = document.getElementById("btn-toggle-calib");
const btnAutoAiSetup = document.getElementById("btn-auto-ai-setup");
const btnAutoCalib = document.getElementById("btn-auto-calib");
const btnClearCalib = document.getElementById("btn-clear-calib");
const modeSelect = document.getElementById("court-mode");
const maxFramesSelect = document.getElementById("max-frames-select");

const videoFileInput = document.getElementById("video-file-input");
const uploadStatusEl = document.getElementById("upload-status");
const uploadProgTrack = document.getElementById("upload-progress-track");
const uploadProgFill = document.getElementById("upload-progress-fill");

const timelineSlider = document.getElementById("timeline-slider");
const btnPlayPause = document.getElementById("btn-play-pause");
const btnSeekBack = document.getElementById("btn-seek-back");
const btnSeekFwd = document.getElementById("btn-seek-fwd");
const btnMute = document.getElementById("btn-mute");
const volumeSlider = document.getElementById("volume-slider");
const chkSfx = document.getElementById("chk-sfx");
const audioStateLabel = document.getElementById("audio-state-label");
const btnSpeed = document.getElementById("btn-speed");
const debugEl = document.getElementById("debug-frame");

const btnScoreA = document.getElementById("btn-score-a");
const btnScoreB = document.getElementById("btn-score-b");
const btnScoreUndo = document.getElementById("btn-score-undo");
const btnScoreReset = document.getElementById("btn-score-reset");
const scoreDisplay = document.getElementById("score-display");
const setDisplay = document.getElementById("set-display");
const dockScoreA = document.getElementById("dock-score-a");
const dockScoreB = document.getElementById("dock-score-b");
const btnDockScoreA = document.getElementById("btn-dock-score-a");
const btnDockScoreB = document.getElementById("btn-dock-score-b");
const scoreHistoryList = document.getElementById("score-history-list");

const heatmapCanvas = document.getElementById("heatmap-canvas");
const bestShotCardEl = document.getElementById("best-shot-card");
const rallySummaryBadgeEl = document.getElementById("rally-summary-badge");
const shotsCountBadgeEl = document.getElementById("shots-count-badge");
const summaryPlayersEl = document.getElementById("summary-players");
const shotsListEl = document.getElementById("shots-list");
const btnDeleteData = document.getElementById("btn-delete-data");

const navHud = document.getElementById("nav-hud");
const navSetup = document.getElementById("nav-setup");
const navSummary = document.getElementById("nav-summary");

const PLAYER_COLORS = ["#22d3ee", "#4ade80", "#c084fc", "#facc15"];

const layerState = {
  topBar: true,
  footRings: true,
  playerTags: true,
  lockOn: true,
  moveTable: true,
  courtMap: true,
};

let matchData = null;
let videoMetaDim = { width: 886, height: 1920 };
let isCalibMode = false;
let isScrubbing = false;
let lastSfxShotFrame = -1;
let lastAutoRallyId = 0;
let manualScoreOverride = false;
let hasUploadedVideo = false;
let draggingHandle = null; // { type: 'calib'|'player', idx: number }

const scoreManager = createScoreManager();
window.__scoreState = scoreManager.state;
window.__currentFrameIdx = 0;
window.__lastHudRenderStats = null;
window.__calibState = { mode: "doubles", points: [], reprojErrorM: 999.0, gridDrawn: false };
window.__assignedPlayers = [];
window.__heatmapDrawn = false;
window.__dataDeleted = false;

function fitStageToVideo() {
  if (window.innerWidth < 900) {
    stageEl.style.width = "390px";
    stageEl.style.height = "844px";
    canvas.width = 390;
    canvas.height = 844;
    calibCanvas.width = 390;
    calibCanvas.height = 844;
    return;
  }
  const rect = stageWrap.getBoundingClientRect();
  const maxW = Math.max(320, rect.width - 12);
  const maxH = Math.max(320, rect.height - 12);
  const vw = video.videoWidth || matchData?.video?.width || videoMetaDim.width || 886;
  const vh = video.videoHeight || matchData?.video?.height || videoMetaDim.height || 1920;
  const scale = Math.min(maxW / vw, maxH / vh);
  const dispW = Math.round(vw * scale);
  const dispH = Math.round(vh * scale);
  stageEl.style.width = `${dispW}px`;
  stageEl.style.height = `${dispH}px`;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(dispW * dpr);
  canvas.height = Math.round(dispH * dpr);
  calibCanvas.width = Math.round(dispW * dpr);
  calibCanvas.height = Math.round(dispH * dpr);
}

window.addEventListener("resize", () => {
  fitStageToVideo();
  if (isCalibMode) renderCalibOverlay();
  else renderFrame(window.__currentFrameIdx || 0);
});

video.addEventListener("loadedmetadata", () => {
  if (video.videoWidth && video.videoHeight) {
    videoMetaDim = { width: video.videoWidth, height: video.videoHeight };
  }
  fitStageToVideo();
  if (isCalibMode) renderCalibOverlay();
  else renderFrame(window.__currentFrameIdx || 0);
});

document.querySelectorAll("#layer-toggles input[type=checkbox]").forEach((cb) => {
  cb.addEventListener("change", (e) => {
    layerState[e.target.dataset.layer] = e.target.checked;
    renderFrame(window.__currentFrameIdx || 0);
  });
});

function formatRallyClock(sec) {
  const mins = Math.floor(sec / 60);
  const rem = Math.max(0, sec - mins * 60);
  const sInt = Math.floor(rem);
  const cs = Math.floor((rem - sInt) * 100);
  return `${String(mins).padStart(2, "0")}:${String(sInt).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

function drawRoundedPanel(c, x, y, w, h, r, fill, stroke, lineW = 1) {
  c.beginPath();
  c.roundRect(x, y, w, h, r);
  if (fill) {
    c.fillStyle = fill;
    c.fill();
  }
  if (stroke) {
    c.strokeStyle = stroke;
    c.lineWidth = lineW;
    c.stroke();
  }
}

// ============================================================================
// ESPORTS BROADCAST HUD RENDERER (Matches Reels Broadcast Reference Video)
// ============================================================================
function renderFrame(frameIdx) {
  if (!matchData || !matchData.frames?.length) return;
  const total = matchData.frames.length;
  const fps = matchData.video.fps;
  const vidFrames = video.duration && Number.isFinite(video.duration) ? Math.round(video.duration * fps) : total;
  const maxSliderIdx = Math.max(total - 1, vidFrames - 1);
  if (timelineSlider && !isScrubbing) {
    timelineSlider.max = String(maxSliderIdx);
    timelineSlider.value = String(Math.min(maxSliderIdx, Math.max(0, frameIdx)));
  }

  const cw = canvas.width;
  const ch = canvas.height;
  if (frameIdx >= total) {
    debugEl.textContent = `frame_idx: ${frameIdx} (${(frameIdx / fps).toFixed(2)}s) · Chưa phân tích`;
    ctx.clearRect(0, 0, cw, ch);
    return;
  }
  const idx = Math.max(0, frameIdx);
  window.__currentFrameIdx = idx;

  const frame = matchData.frames[idx];
  const sx = cw / matchData.video.width;
  const sy = ch / matchData.video.height;
  const u = Math.max(0.78, Math.min(cw / 410, ch / 820));

  debugEl.textContent = `frame_idx: ${idx} (${frame.t.toFixed(2)}s)`;
  ctx.clearRect(0, 0, cw, ch);

  const drawn = {
    rallyTime: false,
    rallyCount: false,
    shotSpeed: false,
    scoreBoard: false,
    lockOn: false,
    playerTags: false,
    footRings: false,
    moveTable: false,
    courtMap: false,
  };

  const playerMap = Object.fromEntries(matchData.players.map((p) => [p.id, p]));
  const activeRally =
    matchData.rallies.find((r) => idx >= r.start_frame && idx <= r.end_frame) || matchData.rallies[0];
  const pastShots = matchData.shots.filter((s) => s.frame <= idx);
  const lastShot = pastShots.length ? pastShots[pastShots.length - 1] : matchData.shots[0];

  // Trigger Web Audio shuttle hit SFX when playback advances across a shot frame
  if (!video.paused && !isScrubbing && chkSfx?.checked && lastShot) {
    if (Math.abs(idx - lastShot.frame) <= 1 && lastShot.frame !== lastSfxShotFrame) {
      lastSfxShotFrame = lastShot.frame;
      playShuttleHitSfx(lastShot.speed_kmh || 180, video.muted ? 0.65 : video.volume);
    }
  }

  // Auto-sync BWF Scoreboard from AI-detected rallies (unless manually overridden by user buttons)
  if (!manualScoreOverride && Array.isArray(matchData.rallies) && matchData.rallies.some((r) => r.winner_team)) {
    const completedRallies = matchData.rallies.filter((r) => r.winner_team && idx >= r.end_frame);
    let autoA = 0;
    let autoB = 0;
    let lastWin = "A";
    for (const cr of completedRallies) {
      if (Array.isArray(cr.score_after) && cr.score_after.length === 2) {
        autoA = cr.score_after[0];
        autoB = cr.score_after[1];
      } else if (cr.winner_team === "A") {
        autoA += 1;
      } else if (cr.winner_team === "B") {
        autoB += 1;
      }
      lastWin = cr.winner_team || lastWin;
    }
    const latestCompleted = completedRallies.length ? completedRallies[completedRallies.length - 1] : null;
    if (latestCompleted && latestCompleted.id !== lastAutoRallyId) {
      lastAutoRallyId = latestCompleted.id;
      if (!video.paused && !isScrubbing && chkSfx?.checked) {
        playScoreChimeSfx(video.muted ? 0.65 : video.volume || 0.8);
      }
    } else if (!latestCompleted) {
      lastAutoRallyId = 0;
    }
    if (scoreManager.state.teamA !== autoA || scoreManager.state.teamB !== autoB) {
      scoreManager.state.teamA = autoA;
      scoreManager.state.teamB = autoB;
      scoreManager.state.serving = lastWin;
      if (scoreDisplay) scoreDisplay.textContent = `${autoA} - ${autoB}`;
      if (setDisplay) setDisplay.textContent = `Set: ${scoreManager.state.setA} - ${scoreManager.state.setB} · Giao cầu: Đội ${lastWin} (Tự động AI)`;
      if (dockScoreA) dockScoreA.textContent = String(autoA);
      if (dockScoreB) dockScoreB.textContent = String(autoB);
    }
  }

  // 1-3. TOP BROADCAST SCOREBOARD (3 distinct glass panels + Minimalist Match Score Pill)
  if (layerState.topBar) {
    ctx.save();
    const pad = Math.round(8 * u);
    const gap = Math.round(6 * u);
    const barY = Math.round(10 * u);
    const barH = Math.round(54 * u);
    const boxW = Math.floor((cw - pad * 2 - gap * 2) / 3);

    const panels = [
      { x: pad, accent: "#22d3ee", label: "RALLY TIME" },
      { x: pad + boxW + gap, accent: "#facc15", label: "RALLY COUNT" },
      { x: pad + (boxW + gap) * 2, accent: "#22d3ee", label: "SHOT SPEED" },
    ];

    for (const p of panels) {
      drawRoundedPanel(ctx, p.x, barY, boxW, barH, 4 * u, "rgba(8, 13, 25, 0.84)", "rgba(148, 163, 184, 0.25)", 1 * u);
      ctx.fillStyle = p.accent;
      ctx.fillRect(p.x, barY + 3 * u, 2.5 * u, barH - 6 * u);
      ctx.fillStyle = "#94a3b8";
      ctx.font = `700 ${Math.round(9.5 * u)}px "Barlow Condensed", sans-serif`;
      ctx.fillText(p.label, p.x + 8 * u, barY + 14 * u);
    }

    const rallyElapsed = activeRally ? Math.max(0, (idx - activeRally.start_frame) / fps) : frame.t;
    ctx.fillStyle = "#ffffff";
    ctx.font = `800 ${Math.round(25 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillText(formatRallyClock(rallyElapsed), panels[0].x + 8 * u, barY + 44 * u);
    drawn.rallyTime = true;

    ctx.fillText(String(pastShots.length), panels[1].x + 8 * u, barY + 44 * u);
    ctx.fillStyle = "#94a3b8";
    ctx.font = `700 ${Math.round(9.5 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillText("SHOT", panels[1].x + boxW - 26 * u, barY + 44 * u);
    drawn.rallyCount = true;

    const spdVal = lastShot && lastShot.speed_kmh ? String(Math.round(lastShot.speed_kmh)) : "--";
    ctx.fillStyle = "#ffffff";
    ctx.font = `800 ${Math.round(25 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillText(spdVal, panels[2].x + 8 * u, barY + 44 * u);
    ctx.fillStyle = "#94a3b8";
    ctx.font = `700 ${Math.round(9.5 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillText("km/h", panels[2].x + boxW - 26 * u, barY + 44 * u);
    drawn.shotSpeed = true;

    // Minimalist Match Score Pill directly below the top bar
    const sc = scoreManager.state;
    const pillW = Math.round(168 * u);
    const pillH = Math.round(22 * u);
    const pillX = Math.round((cw - pillW) / 2);
    const pillY = barY + barH + Math.round(5 * u);
    drawRoundedPanel(ctx, pillX, pillY, pillW, pillH, 11 * u, "rgba(9, 11, 16, 0.86)", "rgba(255, 255, 255, 0.16)", 1 * u);

    ctx.font = `700 ${Math.round(10 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillStyle = "#38bdf8";
    ctx.fillText(`${sc.serving === "A" ? "● " : ""}ĐỘI A`, pillX + 10 * u, pillY + 15 * u);

    const scoreTxt = `${sc.teamA}  :  ${sc.teamB}`;
    ctx.font = `800 ${Math.round(13.5 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillStyle = "#ffffff";
    const sw = ctx.measureText(scoreTxt).width;
    ctx.fillText(scoreTxt, pillX + (pillW - sw) / 2, pillY + 15.5 * u);

    ctx.font = `700 ${Math.round(10 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillStyle = "#facc15";
    const bLabel = `ĐỘI B${sc.serving === "B" ? " ●" : ""}`;
    const bw = ctx.measureText(bLabel).width;
    ctx.fillText(bLabel, pillX + pillW - bw - 10 * u, pillY + 15 * u);
    drawn.scoreBoard = true;

    ctx.restore();
  }

  // 4. GLOWING DOUBLE-RING NEON ELLIPSES UNDER FEET (via inverse Homography)
  if (layerState.footRings) {
    for (const p of matchData.players) {
      const pf = frame.players[p.id];
      if (!pf || !pf.foot_court_m || (pf.conf !== undefined && pf.conf <= 0.1)) continue;
      const outerPts = projectFootRingPx(
        matchData.court.homography_img_to_court,
        pf.foot_court_m[0],
        pf.foot_court_m[1],
        0.52,
        32
      );
      const innerPts = projectFootRingPx(
        matchData.court.homography_img_to_court,
        pf.foot_court_m[0],
        pf.foot_court_m[1],
        0.37,
        32
      );

      // Outer glowing neon ring
      ctx.save();
      ctx.beginPath();
      outerPts.forEach(([px, py], k) => {
        if (k === 0) ctx.moveTo(px * sx, py * sy);
        else ctx.lineTo(px * sx, py * sy);
      });
      ctx.closePath();
      ctx.fillStyle = p.color + "1c";
      ctx.fill();
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 12 * u;
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 2.6 * u;
      ctx.stroke();

      // White-hot leading arc highlight (like the Reels reference ring!)
      ctx.beginPath();
      for (let k = 0; k <= 10; k++) {
        const [px, py] = outerPts[k];
        if (k === 0) ctx.moveTo(px * sx, py * sy);
        else ctx.lineTo(px * sx, py * sy);
      }
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.2 * u;
      ctx.stroke();

      // Inner concentric thin ring
      ctx.shadowBlur = 0;
      ctx.beginPath();
      innerPts.forEach(([px, py], k) => {
        if (k === 0) ctx.moveTo(px * sx, py * sy);
        else ctx.lineTo(px * sx, py * sy);
      });
      ctx.closePath();
      ctx.strokeStyle = p.color + "aa";
      ctx.lineWidth = 1.2 * u;
      ctx.stroke();
      ctx.restore();
    }
    drawn.footRings = true;
  }

  // 5. FLOATING ESPORTS NAMEPLATES ABOVE HEAD (Name + Energy + Glowing Bar + Pointer ▼)
  if (layerState.playerTags) {
    ctx.save();
    for (const p of matchData.players) {
      const pf = frame.players[p.id];
      if (!pf || !pf.bbox_px || (pf.conf !== undefined && pf.conf <= 0.1)) continue;
      const [bx1, by1, bx2] = pf.bbox_px;
      const cx = ((bx1 + bx2) / 2) * sx;
      const tagW = Math.round(74 * u);
      const tagH = Math.round(22 * u);
      const topY = Math.max(70 * u, by1 * sy - tagH - 9 * u);
      const leftX = Math.max(4, Math.min(cw - tagW - 4, cx - tagW / 2));

      drawRoundedPanel(ctx, leftX, topY, tagW, tagH, 3 * u, "rgba(9, 15, 28, 0.88)", "rgba(148, 163, 184, 0.35)", 1 * u);

      ctx.fillStyle = "#f8fafc";
      ctx.font = `700 ${Math.round(10 * u)}px "Plus Jakarta Sans", sans-serif`;
      ctx.fillText(p.name.slice(0, 7), leftX + 5 * u, topY + 12.5 * u);

      const eTxt = String(Math.round(pf.energy));
      ctx.font = `800 ${Math.round(11.5 * u)}px "Barlow Condensed", sans-serif`;
      const tw = ctx.measureText(eTxt).width;
      ctx.fillStyle = "#ffffff";
      ctx.fillText(eTxt, leftX + tagW - tw - 5 * u, topY + 12.5 * u);

      // Bottom glowing energy bar
      const barPad = 4 * u;
      const barMaxW = tagW - barPad * 2;
      ctx.fillStyle = "rgba(51, 65, 85, 0.8)";
      ctx.fillRect(leftX + barPad, topY + tagH - 4.5 * u, barMaxW, 2.8 * u);
      ctx.fillStyle = p.color;
      ctx.fillRect(leftX + barPad, topY + tagH - 4.5 * u, barMaxW * Math.max(0, Math.min(1, pf.energy / 100)), 2.8 * u);

      // Small downward pointer triangle ▼
      ctx.beginPath();
      ctx.moveTo(cx - 4 * u, topY + tagH + 1 * u);
      ctx.lineTo(cx + 4 * u, topY + tagH + 1 * u);
      ctx.lineTo(cx, topY + tagH + 6 * u);
      ctx.closePath();
      ctx.fillStyle = p.color;
      ctx.fill();
    }
    drawn.playerTags = true;
    ctx.restore();
  }

  // 6. LOCK ON SHUTTLE RETICLE ([⌜ ⌝ ⌞ ⌟] + Inner Circle + Tapered Comet Trail + Flag)
  if (layerState.lockOn && frame.shuttle && frame.shuttle.visible && frame.shuttle.x_px != null) {
    ctx.save();
    const trailStart = Math.max(0, idx - 9);
    const trailPts = [];
    for (let k = trailStart; k <= idx; k++) {
      const sf = matchData.frames[k].shuttle;
      if (sf && sf.visible && sf.x_px != null) {
        trailPts.push([sf.x_px * sx, sf.y_px * sy]);
      }
    }
    // Tapered golden comet trail
    for (let k = 1; k < trailPts.length; k++) {
      const alpha = k / trailPts.length;
      ctx.beginPath();
      ctx.moveTo(trailPts[k - 1][0], trailPts[k - 1][1]);
      ctx.lineTo(trailPts[k][0], trailPts[k][1]);
      ctx.strokeStyle = `rgba(250, 204, 21, ${(0.18 + 0.75 * alpha).toFixed(2)})`;
      ctx.lineWidth = (1.0 + 2.4 * alpha) * u;
      ctx.stroke();
    }

    const sxPx = frame.shuttle.x_px * sx;
    const syPx = frame.shuttle.y_px * sy;
    const r = 15 * u;
    const cLen = 5.5 * u;

    // 4 Corner brackets [⌜ ⌝ ⌞ ⌟]
    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 2.0 * u;
    const corners = [
      [sxPx - r, syPx - r, 1, 1],
      [sxPx + r, syPx - r, -1, 1],
      [sxPx - r, syPx + r, 1, -1],
      [sxPx + r, syPx + r, -1, -1],
    ];
    for (const [bx, by, dx, dy] of corners) {
      ctx.beginPath();
      ctx.moveTo(bx + dx * cLen, by);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx, by + dy * cLen);
      ctx.stroke();
    }

    // Broken inner targeting circle
    ctx.beginPath();
    ctx.arc(sxPx, syPx, 8 * u, -0.3, Math.PI * 0.8);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(sxPx, syPx, 8 * u, Math.PI * 1.05, Math.PI * 1.85);
    ctx.stroke();

    // Callout banner with left vertical gold accent bar
    const hitter = lastShot ? playerMap[lastShot.player_id]?.name || lastShot.player_id : "";
    const spdLabel = lastShot && lastShot.speed_kmh ? `${Math.round(lastShot.speed_kmh)} km/h` : "-- km/h";
    const flagW = 98 * u;
    const flagH = 28 * u;
    const fx = sxPx + flagW + 22 * u < cw ? sxPx + r + 5 * u : sxPx - r - flagW - 5 * u;
    const fy = Math.max(70 * u, syPx - flagH - 2 * u);

    ctx.fillStyle = "rgba(10, 15, 28, 0.86)";
    ctx.fillRect(fx, fy, flagW, flagH);
    ctx.fillStyle = "#facc15";
    ctx.fillRect(fx, fy, 2.5 * u, flagH);

    ctx.font = `700 ${Math.round(8.5 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillStyle = "#facc15";
    ctx.fillText("LOCK ON", fx + 6 * u, fy + 10 * u);

    ctx.fillStyle = "#cbd5e1";
    ctx.font = `600 ${Math.round(8.5 * u)}px "Plus Jakarta Sans", sans-serif`;
    ctx.fillText(hitter.slice(0, 7), fx + 54 * u, fy + 10 * u);

    ctx.fillStyle = "#ffffff";
    ctx.font = `800 ${Math.round(14 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillText(spdLabel, fx + 6 * u, fy + 24 * u);

    drawn.lockOn = true;
    ctx.restore();
  }

  // Bottom HUD Band Geometry (reserve bottom controls bar height on mobile)
  const controlsBarNode = document.querySelector(".controls-bar");
  const controlsHeight = controlsBarNode ? controlsBarNode.offsetHeight : 112;
  const bottomReserve = window.innerWidth < 900 ? Math.max(118, controlsHeight + 10) : 14;
  const bandH = Math.round(148 * u);
  const bandY = ch - bandH - bottomReserve;
  const padX = Math.round(8 * u);
  const leftW = Math.round((cw - padX * 3) * 0.51);
  const rightW = cw - padX * 3 - leftW;

  // 7. STACKED PLAYER TELEMETRY STRIPS WITH SEGMENTED LED MOVE BAR (Bottom-Left)
  if (layerState.moveTable) {
    ctx.save();
    const rows = matchData.players.slice(0, 4);
    const gapY = 3 * u;
    const stripH = (bandH - gapY * (Math.max(1, rows.length) - 1)) / Math.max(1, rows.length);

    rows.forEach((p, rIdx) => {
      const pf = frame.players[p.id];
      const sy0 = bandY + rIdx * (stripH + gapY);
      drawRoundedPanel(ctx, padX, sy0, leftW, stripH, 3 * u, "rgba(8, 13, 25, 0.86)", "rgba(148, 163, 184, 0.22)", 1 * u);

      // Colored top separator line
      ctx.fillStyle = p.color;
      ctx.fillRect(padX, sy0 + stripH * 0.48, leftW, 1.2 * u);

      // Top half: Name (colored) | ENERGY label | Score
      ctx.fillStyle = p.color;
      ctx.font = `700 ${Math.round(10.5 * u)}px "Plus Jakarta Sans", sans-serif`;
      ctx.fillText(p.name.slice(0, 9), padX + 5 * u, sy0 + stripH * 0.37);

      const eNum = String(Math.round(pf?.energy ?? 0));
      ctx.fillStyle = "#ffffff";
      ctx.font = `800 ${Math.round(14 * u)}px "Barlow Condensed", sans-serif`;
      const ew = ctx.measureText(eNum).width;
      ctx.fillText(eNum, padX + leftW - ew - 5 * u, sy0 + stripH * 0.39);

      ctx.fillStyle = "#94a3b8";
      ctx.font = `600 ${Math.round(8 * u)}px "Plus Jakarta Sans", sans-serif`;
      ctx.fillText("ENERGY", padX + leftW - ew - 38 * u, sy0 + stripH * 0.36);

      // Bottom half: MOVE | 18 segmented LED blocks | xx.x m
      ctx.fillStyle = "#94a3b8";
      ctx.font = `700 ${Math.round(8 * u)}px "Barlow Condensed", sans-serif`;
      ctx.fillText("MOVE", padX + 5 * u, sy0 + stripH * 0.82);

      const distVal = pf?.dist_cum_m ?? 0;
      const distTxt = `${distVal.toFixed(1)} m`;
      ctx.fillStyle = "#e2e8f0";
      ctx.font = `700 ${Math.round(9.5 * u)}px "Barlow Condensed", sans-serif`;
      const dw = ctx.measureText(distTxt).width;
      ctx.fillText(distTxt, padX + leftW - dw - 5 * u, sy0 + stripH * 0.83);

      // Segmented LED bar
      const segStartX = padX + 26 * u;
      const segAvailW = Math.max(24 * u, leftW - 34 * u - dw);
      const nSegs = 18;
      const segGap = 1.6 * u;
      const segW = (segAvailW - segGap * (nSegs - 1)) / nSegs;
      const activeSegs = Math.round(((pf?.energy ?? 0) / 100) * nSegs);
      for (let s = 0; s < nSegs; s++) {
        ctx.fillStyle = s < activeSegs ? p.color : "rgba(51, 65, 85, 0.7)";
        ctx.fillRect(segStartX + s * (segW + segGap), sy0 + stripH * 0.63, segW, stripH * 0.22);
      }
    });
    drawn.moveTable = true;
    ctx.restore();
  }

  // 8. HORIZONTAL 2D COURT MAP RADAR (Bottom-Right, matches Reels Reference Video!)
  if (layerState.courtMap) {
    ctx.save();
    const mapX = padX * 2 + leftW;
    const mapY = bandY;
    drawRoundedPanel(ctx, mapX, mapY, rightW, bandH, 4 * u, "rgba(8, 13, 25, 0.88)", "rgba(148, 163, 184, 0.28)", 1 * u);

    ctx.fillStyle = "#f8fafc";
    ctx.font = `700 ${Math.round(10 * u)}px "Barlow Condensed", sans-serif`;
    ctx.fillText("COURT MAP", mapX + 8 * u, mapY + 14 * u);

    const cPadX = 8 * u;
    const cTop = mapY + 20 * u;
    const cH = bandH - 36 * u;
    const cW = rightW - cPadX * 2;
    const cX = mapX + cPadX;

    // Court floor & lines (Horizontal orientation: length 13.4m left->right, width 6.1m top->bottom)
    ctx.fillStyle = "rgba(15, 28, 46, 0.85)";
    ctx.fillRect(cX, cTop, cW, cH);
    ctx.strokeStyle = "rgba(203, 213, 225, 0.65)";
    ctx.lineWidth = 1 * u;
    ctx.strokeRect(cX, cTop, cW, cH);

    // Doubles alleys + short service lines + center lines
    const alleyY = (0.46 / 6.1) * cH;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
    ctx.beginPath();
    ctx.moveTo(cX, cTop + alleyY);
    ctx.lineTo(cX + cW, cTop + alleyY);
    ctx.moveTo(cX, cTop + cH - alleyY);
    ctx.lineTo(cX + cW, cTop + cH - alleyY);
    const sslL = cX + ((6.7 - 1.98) / 13.4) * cW;
    const sslR = cX + ((6.7 + 1.98) / 13.4) * cW;
    ctx.moveTo(sslL, cTop);
    ctx.lineTo(sslL, cTop + cH);
    ctx.moveTo(sslR, cTop);
    ctx.lineTo(sslR, cTop + cH);
    ctx.moveTo(cX, cTop + cH / 2);
    ctx.lineTo(sslL, cTop + cH / 2);
    ctx.moveTo(sslR, cTop + cH / 2);
    ctx.lineTo(cX + cW, cTop + cH / 2);
    ctx.stroke();

    // Vertical dashed net line at center (Y = 6.7m)
    const netX = cX + 0.5 * cW;
    ctx.save();
    ctx.setLineDash([3 * u, 3 * u]);
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1.3 * u;
    ctx.beginPath();
    ctx.moveTo(netX, cTop - 2 * u);
    ctx.lineTo(netX, cTop + cH + 2 * u);
    ctx.stroke();
    ctx.restore();

    // Map court_m [x_m in 0..w, y_m in 0..13.4]: Team A (y > 6.7) on left half, Team B (y < 6.7) on right half
    const courtW_m = matchData.court.dims_m.width || 6.1;
    const toRadarXY = (xm, ym) => {
      const rx = cX + ((13.4 - ym) / 13.4) * cW;
      const ry = cTop + (xm / courtW_m) * cH;
      return [
        Math.max(cX + 3 * u, Math.min(cX + cW - 3 * u, rx)),
        Math.max(cTop + 3 * u, Math.min(cTop + cH - 3 * u, ry)),
      ];
    };

    // 3-second trails + glowing player dots
    const trailFrames = Math.round(3.0 * fps);
    const tStart = Math.max(0, idx - trailFrames);
    for (const p of matchData.players) {
      ctx.beginPath();
      let started = false;
      for (let k = tStart; k <= idx; k++) {
        const pf = matchData.frames[k].players[p.id];
        if (!pf || !pf.foot_court_m) continue;
        const [rx, ry] = toRadarXY(pf.foot_court_m[0], pf.foot_court_m[1]);
        if (!started) {
          ctx.moveTo(rx, ry);
          started = true;
        } else {
          ctx.lineTo(rx, ry);
        }
      }
      ctx.strokeStyle = p.color + "aa";
      ctx.lineWidth = 1.6 * u;
      ctx.stroke();

      const cur = frame.players[p.id];
      if (cur && cur.foot_court_m) {
        const [rx, ry] = toRadarXY(cur.foot_court_m[0], cur.foot_court_m[1]);
        ctx.beginPath();
        ctx.arc(rx, ry, 4.0 * u, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.1 * u;
        ctx.stroke();
      }
    }

    ctx.fillStyle = "#94a3b8";
    ctx.font = `600 ${Math.round(8.5 * u)}px "Plus Jakarta Sans", sans-serif`;
    ctx.fillText("Đội mình", cX, mapY + bandH - 5 * u);
    const oppTxt = "Đối thủ";
    const ow = ctx.measureText(oppTxt).width;
    ctx.fillText(oppTxt, cX + cW - ow, mapY + bandH - 5 * u);

    drawn.courtMap = true;
    ctx.restore();
  }

  window.__lastHudRenderStats = { frameIdx: idx, t: frame.t, drawn };
}

// ============================================================================
// FRAME-ACCURATE VIDEO SYNC LOOP
// ============================================================================
function syncFromMediaTime(mediaTime) {
  if (!matchData) return;
  const frameIdx = Math.round(mediaTime * matchData.video.fps);
  if (!isCalibMode) {
    renderFrame(frameIdx);
  }
}

function startSyncLoop() {
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const onFrame = (_now, metadata) => {
      syncFromMediaTime(metadata.mediaTime);
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    const onRaf = () => {
      syncFromMediaTime(video.currentTime);
      requestAnimationFrame(onRaf);
    };
    requestAnimationFrame(onRaf);
  }
  video.addEventListener("seeked", () => {
    syncFromMediaTime(video.currentTime);
  });
}

window.__seekToTime = async (targetSec) => {
  if (!matchData) throw new Error("Match data not loaded");
  if (video.readyState < 1) {
    await new Promise((res) => {
      const done = () => {
        video.removeEventListener("loadedmetadata", done);
        video.removeEventListener("error", done);
        res();
      };
      video.addEventListener("loadedmetadata", done);
      video.addEventListener("error", done);
      setTimeout(done, 1200);
    });
  }
  await new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      video.removeEventListener("seeked", finish);
      syncFromMediaTime(targetSec);
      resolve();
    };
    video.addEventListener("seeked", finish);
    video.currentTime = targetSec;
    setTimeout(finish, 800);
  });
};

// ============================================================================
// ON-VIDEO COURT CALIBRATION & DRAGGABLE HANDLES + PLAYER TAGGING
// ============================================================================
function getVideoCoordSpace() {
  return {
    vw: videoMetaDim.width || matchData?.video?.width || 886,
    vh: videoMetaDim.height || matchData?.video?.height || 1920,
  };
}

function getTargetCourtCoords(mode) {
  const w = mode === "doubles" ? 6.1 : 5.18;
  return [
    [0.0, 0.0],
    [w, 0.0],
    [0.0, 6.7],
    [w, 6.7],
    [0.0, 13.4],
    [w, 13.4],
  ];
}

function renderPlayerEditRows() {
  if (!playerEditListEl) return;
  playerEditListEl.innerHTML = "";
  window.__assignedPlayers.forEach((ap, idx) => {
    const row = document.createElement("div");
    row.className = "player-edit-row";
    const col = PLAYER_COLORS[idx % PLAYER_COLORS.length];
    row.innerHTML = `
      <span class="color-dot" style="background:${col};"></span>
      <input type="text" value="${ap.name}" data-pidx="${idx}" style="flex:1; min-width:0;" />
      <select data-pteam="${idx}">
        <option value="A" ${ap.team === "A" ? "selected" : ""}>Đội A (Gần)</option>
        <option value="B" ${ap.team === "B" ? "selected" : ""}>Đội B (Xa)</option>
      </select>
    `;
    row.querySelector("input").addEventListener("input", (e) => {
      window.__assignedPlayers[idx].name = e.target.value;
      renderCalibOverlay();
    });
    row.querySelector("select").addEventListener("change", (e) => {
      window.__assignedPlayers[idx].team = e.target.value;
      renderCalibOverlay();
    });
    playerEditListEl.appendChild(row);
  });
}

function renderCalibOverlay() {
  const { vw, vh } = getVideoCoordSpace();
  const sx = calibCanvas.width / vw;
  const sy = calibCanvas.height / vh;
  const u = Math.max(0.85, Math.min(calibCanvas.width / 410, calibCanvas.height / 820));
  calibCtx.clearRect(0, 0, calibCanvas.width, calibCanvas.height);

  const pts = window.__calibState.points;
  const mode = modeSelect.value;
  window.__calibState.mode = mode;
  const wM = mode === "doubles" ? 6.1 : 5.18;

  if (pts.length >= 6) {
    const { H, reprojErrorM } = solveCourtCalibration(pts);
    window.__calibState.reprojErrorM = reprojErrorM;
    window.__calibState.gridDrawn = true;

    const lines = [
      [[0, 0], [wM, 0]],
      [[wM, 0], [wM, 13.4]],
      [[wM, 13.4], [0, 13.4]],
      [[0, 13.4], [0, 0]],
      [[0, 6.7], [wM, 6.7]],
      [[0, 4.72], [wM, 4.72]],
      [[0, 8.68], [wM, 8.68]],
      [[wM / 2, 0], [wM / 2, 4.72]],
      [[wM / 2, 8.68], [wM / 2, 13.4]],
    ];

    // Subtle translucent court fill
    calibCtx.save();
    const cCorners = [[0, 0], [wM, 0], [wM, 13.4], [0, 13.4]].map(([xm, ym]) => courtToPx(H, xm, ym));
    calibCtx.beginPath();
    cCorners.forEach(([px, py], i) => {
      if (i === 0) calibCtx.moveTo(px * sx, py * sy);
      else calibCtx.lineTo(px * sx, py * sy);
    });
    calibCtx.closePath();
    calibCtx.fillStyle = reprojErrorM <= 0.25 ? "rgba(34, 211, 238, 0.12)" : "rgba(244, 63, 94, 0.15)";
    calibCtx.fill();

    for (const [[x1, y1], [x2, y2]] of lines) {
      const [px1, py1] = courtToPx(H, x1, y1);
      const [px2, py2] = courtToPx(H, x2, y2);
      calibCtx.beginPath();
      calibCtx.moveTo(px1 * sx, py1 * sy);
      calibCtx.lineTo(px2 * sx, py2 * sy);
      const isNet = y1 === 6.7 && y2 === 6.7;
      calibCtx.strokeStyle = reprojErrorM > 0.25 ? "#f43f5e" : isNet ? "#facc15" : "#22d3ee";
      calibCtx.lineWidth = (isNet ? 2.6 : 2.0) * u;
      calibCtx.stroke();
    }
    calibCtx.restore();

    const locked = reprojErrorM > 0.25;
    btnProcess.disabled = locked;
    calibStatusEl.textContent = `Điểm: ${pts.length}/6 | Sai số chiếu: ${reprojErrorM.toFixed(2)}m ${locked ? "(Kéo lại P3/P4 cho thẳng vạch lưới!)" : "(Chuẩn — Sẵn sàng xử lý)"
      }`;
    calibStatusEl.style.color = locked ? "#f43f5e" : "#4ade80";
  } else {
    window.__calibState.reprojErrorM = 999.0;
    window.__calibState.gridDrawn = false;
    btnProcess.disabled = true;
    const labels = ["Góc Xa-Trái", "Góc Xa-Phải", "Cọc Lưới Trái", "Cọc Lưới Phải", "Góc Gần-Trái", "Góc Gần-Phải"];
    calibStatusEl.textContent = `Điểm: ${pts.length}/6 — Bấm lên video để chấm: ${labels[pts.length]}`;
    calibStatusEl.style.color = "#facc15";
  }

  // Draw draggable court corner handles P1..P6
  const ptNames = ["P1 Xa-Trái", "P2 Xa-Phải", "P3 Lưới-Trái", "P4 Lưới-Phải", "P5 Gần-Trái", "P6 Gần-Phải"];
  pts.forEach((p, idx) => {
    const cx = p.px[0] * sx;
    const cy = p.px[1] * sy;
    calibCtx.beginPath();
    calibCtx.arc(cx, cy, 7.5 * u, 0, Math.PI * 2);
    calibCtx.fillStyle = "#facc15";
    calibCtx.fill();
    calibCtx.strokeStyle = "#090d16";
    calibCtx.lineWidth = 2 * u;
    calibCtx.stroke();

    drawRoundedPanel(calibCtx, cx + 10 * u, cy - 10 * u, 68 * u, 17 * u, 3 * u, "rgba(9, 13, 22, 0.85)", "#facc15", 1);
    calibCtx.fillStyle = "#facc15";
    calibCtx.font = `700 ${Math.round(9.5 * u)}px "Plus Jakarta Sans", sans-serif`;
    calibCtx.fillText(ptNames[idx] || `P${idx + 1}`, cx + 14 * u, cy + 2 * u);
  });

  // Draw assigned player markers on video
  window.__assignedPlayers.forEach((ap, idx) => {
    const cx = ap.px[0] * sx;
    const cy = ap.px[1] * sy;
    const col = PLAYER_COLORS[idx % PLAYER_COLORS.length];

    calibCtx.beginPath();
    calibCtx.arc(cx, cy, 8 * u, 0, Math.PI * 2);
    calibCtx.fillStyle = col;
    calibCtx.fill();
    calibCtx.strokeStyle = "#ffffff";
    calibCtx.lineWidth = 2 * u;
    calibCtx.stroke();

    drawRoundedPanel(calibCtx, cx - 36 * u, cy - 28 * u, 72 * u, 18 * u, 4 * u, "rgba(9, 13, 22, 0.9)", col, 1.2 * u);
    calibCtx.fillStyle = "#ffffff";
    calibCtx.font = `700 ${Math.round(9.5 * u)}px "Plus Jakarta Sans", sans-serif`;
    calibCtx.fillText(`${ap.name} (${ap.team})`, cx - 31 * u, cy - 16 * u);
  });

  assignStatusEl.textContent = `Đã gán: ${window.__assignedPlayers.length} người chơi`;
}

function setCalibOverlayActive(active) {
  isCalibMode = active;
  calibCanvas.style.display = active ? "block" : "none";
  calibBanner.style.display = active ? "block" : "none";
  canvas.style.display = active ? "none" : "block";
  if (btnToggleCalib) {
    btnToggleCalib.textContent = active ? "Xem HUD" : "Hiện lưới sân";
  }
  if (modePill) {
    modePill.textContent = active ? "ĐỊNH HÌNH SÂN" : "HUD LIVE";
  }
  if (active) {
    video.pause();
    if (btnPlayPause) btnPlayPause.textContent = "▶ Phát";
    renderCalibOverlay();
  } else {
    renderFrame(window.__currentFrameIdx || 0);
  }
}

window.__addCalibPointPx = (px, py) => {
  const targets = getTargetCourtCoords(modeSelect.value);
  const idx = window.__calibState.points.length;
  if (idx < 6) {
    window.__calibState.points.push({ px: [px, py], court_m: targets[idx] });
    setCalibOverlayActive(true);
  }
};

window.__moveCalibPointPx = (idx, px, py) => {
  if (window.__calibState.points[idx]) {
    window.__calibState.points[idx].px = [px, py];
    renderCalibOverlay();
  }
};

window.__resetCalibValid = () => {
  const { vw, vh } = getVideoCoordSpace();
  window.__calibState.points = [];
  if (vw === 886 && vh === 1920) {
    for (const [px, py] of [
      [230, 760],
      [656, 760],
      [171.84, 1014.44],
      [714.16, 1014.44],
      [70, 1460],
      [816, 1460],
    ]) {
      window.__addCalibPointPx(px, py);
    }
  } else {
    // Compute projectively-consistent 6 points for any video resolution
    const p1 = [vw * 0.26, vh * 0.44];
    const p2 = [vw * 0.74, vh * 0.44];
    const p5 = [vw * 0.08, vh * 0.88];
    const p6 = [vw * 0.92, vh * 0.88];
    const tmp = [
      { px: p1, court_m: [0, 0] },
      { px: p2, court_m: [6.1, 0] },
      { px: [0, 0], court_m: [0, 6.7] },
      { px: [0, 0], court_m: [6.1, 6.7] },
      { px: p5, court_m: [0, 13.4] },
      { px: p6, court_m: [6.1, 13.4] },
    ];
    const { H } = solveCourtCalibration(tmp);
    const w = modeSelect.value === "doubles" ? 6.1 : 5.18;
    const p3 = courtToPx(H, 0.0, 6.7);
    const p4 = courtToPx(H, w, 6.7);
    for (const pt of [p1, p2, p3, p4, p5, p6]) {
      window.__addCalibPointPx(pt[0], pt[1]);
    }
  }
};

window.__assignPlayerTap = (px, py, name, team) => {
  const id = `p${window.__assignedPlayers.length + 1}`;
  window.__assignedPlayers.push({ id, name, team, px: [px, py] });
  renderPlayerEditRows();
  renderCalibOverlay();
};

// Pointer drag-and-drop on #calib-canvas (drag P1..P6 or player markers, or click to add)
function eventToVideoPx(e) {
  const rect = calibCanvas.getBoundingClientRect();
  const { vw, vh } = getVideoCoordSpace();
  return [
    ((e.clientX - rect.left) / rect.width) * vw,
    ((e.clientY - rect.top) / rect.height) * vh,
  ];
}

calibCanvas.addEventListener("pointerdown", (e) => {
  const [px, py] = eventToVideoPx(e);
  const { vw, vh } = getVideoCoordSpace();
  const hitRadius = Math.max(vw, vh) * 0.032;

  // Check if grabbing an existing P1..P6 handle
  for (let i = 0; i < window.__calibState.points.length; i++) {
    const [hx, hy] = window.__calibState.points[i].px;
    if (Math.hypot(px - hx, py - hy) <= hitRadius) {
      draggingHandle = { type: "calib", idx: i };
      calibCanvas.setPointerCapture(e.pointerId);
      return;
    }
  }
  // Check if grabbing an existing player marker
  for (let i = 0; i < window.__assignedPlayers.length; i++) {
    const [hx, hy] = window.__assignedPlayers[i].px;
    if (Math.hypot(px - hx, py - hy) <= hitRadius) {
      draggingHandle = { type: "player", idx: i };
      calibCanvas.setPointerCapture(e.pointerId);
      return;
    }
  }

  // Otherwise add calibration point or player marker
  if (window.__calibState.points.length < 6) {
    window.__addCalibPointPx(px, py);
    // Auto-derive exact projective net points if user placed 4 corners, or let them place all 6
  } else if (window.__assignedPlayers.length < 4) {
    const idx = window.__assignedPlayers.length + 1;
    const team = py > vh * 0.54 ? "A" : "B";
    const defaultNames = ["Foam", "Blue", "Opp 1", "Kotty"];
    window.__assignPlayerTap(px, py, defaultNames[idx - 1] || `Player ${idx}`, team);
  }
});

calibCanvas.addEventListener("pointermove", (e) => {
  if (!draggingHandle) return;
  const [px, py] = eventToVideoPx(e);
  if (draggingHandle.type === "calib") {
    const idx = draggingHandle.idx;
    window.__calibState.points[idx].px = [px, py];
    // If dragging one of the 4 outer corners (0, 1, 4, 5) while all 6 points exist, keep P3/P4 snapped on the projective net plane!
    if (window.__calibState.points.length === 6 && (idx === 0 || idx === 1 || idx === 4 || idx === 5)) {
      const { H } = solveCourtCalibration(window.__calibState.points);
      const w = modeSelect.value === "doubles" ? 6.1 : 5.18;
      window.__calibState.points[2].px = courtToPx(H, 0.0, 6.7);
      window.__calibState.points[3].px = courtToPx(H, w, 6.7);
    }
    renderCalibOverlay();
  } else if (draggingHandle.type === "player") {
    window.__assignedPlayers[draggingHandle.idx].px = [px, py];
    renderCalibOverlay();
  }
});

calibCanvas.addEventListener("pointerup", () => {
  draggingHandle = null;
});

// ============================================================================
// SUMMARY & HEATMAP PANEL
// ============================================================================
const SHOT_TYPE_VN = {
  Smash: { label: "Đập cầu", color: "#f43f5e", bg: "rgba(244, 63, 94, 0.18)" },
  Drive: { label: "Phản tạt", color: "#38bdf8", bg: "rgba(56, 189, 248, 0.16)" },
  Clear: { label: "Phông cầu", color: "#a78bfa", bg: "rgba(167, 139, 250, 0.18)" },
  Drop: { label: "Bỏ nhỏ", color: "#34d399", bg: "rgba(52, 211, 153, 0.16)" },
  Serve: { label: "Giao cầu", color: "#facc15", bg: "rgba(250, 204, 21, 0.16)" },
  Lift: { label: "Phòng thủ", color: "#94a3b8", bg: "rgba(148, 163, 184, 0.18)" },
};

function inferShotTypeClient(shot, idx) {
  if (shot.shot_type && SHOT_TYPE_VN[shot.shot_type]) return shot.shot_type;
  if (idx === 0 && (!shot.speed_kmh || shot.speed_kmh < 105)) return "Serve";
  const spd = shot.speed_kmh || 0;
  if (spd >= 115) return "Smash";
  const distNet = shot.to_court_m ? Math.abs(shot.to_court_m[1] - 6.7) : 3.0;
  if (distNet <= 2.2 && spd < 90) return "Drop";
  if (distNet >= 4.0) return "Clear";
  return "Drive";
}

function evaluatePlayerStyle(summaryItem, playerShots) {
  let maxShotKmh = 0;
  let smashCount = 0;
  for (let i = 0; i < playerShots.length; i++) {
    const s = playerShots[i];
    if (s.speed_kmh && s.speed_kmh > maxShotKmh) maxShotKmh = s.speed_kmh;
    if (inferShotTypeClient(s, i) === "Smash") smashCount++;
  }
  const front = summaryItem?.zones_pct?.front || 0;
  const back = summaryItem?.zones_pct?.back || 0;
  const dist = summaryItem?.total_distance_m || 0;

  if (maxShotKmh >= 120 || smashCount >= 2) {
    return { badge: "🔥 Tay đập chủ lực", desc: "Tấn công uy lực từ cuối sân, tốc độ cầu cao." };
  }
  if (front >= 38) {
    return { badge: "🎯 Kiểm soát trên lưới", desc: "Chủ động bắt lưới, bỏ nhỏ và phản tạt nhanh." };
  }
  if (back >= 45) {
    return { badge: "🛡️ Điều cầu cuối sân", desc: "Bao quát phần sân sau, phông cầu và đập cầu." };
  }
  if (dist >= 16) {
    return { badge: "⚡ Bao sân bền bỉ", desc: "Di chuyển rộng khắp mặt sân, thể lực dồi dào." };
  }
  return { badge: "⚖️ Công thủ toàn diện", desc: "Phân bổ vị trí cân bằng giữa công và thủ." };
}

function renderSummaryPanel() {
  if (!matchData) return;
  const playerMap = Object.fromEntries(matchData.players.map((p) => [p.id, p]));
  const fps = matchData.video.fps || 30;

  // 1. Find Best Shot (Top Speed)
  let bestShot = null;
  for (const s of matchData.shots) {
    if (s.speed_kmh != null && (!bestShot || s.speed_kmh > bestShot.speed_kmh)) {
      bestShot = s;
    }
  }
  if (!bestShot && matchData.shots.length > 0) {
    bestShot = matchData.shots[0];
  }

  if (shotsCountBadgeEl) {
    shotsCountBadgeEl.textContent = `${matchData.shots.length} cú chạm · ${matchData.rallies.length} pha cầu`;
  }
  if (rallySummaryBadgeEl) {
    const lastR = matchData.rallies[matchData.rallies.length - 1];
    if (lastR && Array.isArray(lastR.score_after)) {
      rallySummaryBadgeEl.textContent = `Tỉ số AI: Đội A ${lastR.score_after[0]} - ${lastR.score_after[1]} Đội B`;
    } else {
      rallySummaryBadgeEl.textContent = `${matchData.rallies.length} Rally`;
    }
  }

  if (bestShotCardEl) {
    if (bestShot) {
      const pInfo = playerMap[bestShot.player_id] || { name: bestShot.player_id, color: "#38bdf8", team: "A" };
      const stKey = inferShotTypeClient(bestShot, bestShot.idx - 1);
      const stMeta = SHOT_TYPE_VN[stKey] || SHOT_TYPE_VN.Drive;
      const tSec = (bestShot.frame / fps).toFixed(2);
      bestShotCardEl.innerHTML = `
        <div class="best-shot-banner">
          <div>
            <div style="font-size:11px; color:#e2e8f0; font-weight:600;">
              <span style="color:${pInfo.color}; font-weight:700;">${pInfo.name}</span> (Đội ${pInfo.team})
              <span class="shot-badge" style="color:${stMeta.color}; background:${stMeta.bg};">${stMeta.label}</span>
            </div>
            <div style="font-size:10.5px; color:var(--text-muted); margin-top:3px;">
              Cú chạm #${bestShot.idx} ở giây ${tSec}s
            </div>
            <button class="btn primary" id="btn-jump-best-shot" type="button" style="margin-top:6px; padding:4px 9px; font-size:11px;">
              ▶ Xem pha cầu này
            </button>
          </div>
          <div style="text-align:right;">
            <div class="best-shot-speed">${bestShot.speed_kmh ? Math.round(bestShot.speed_kmh) : "--"}</div>
            <div style="font-size:10px; color:var(--gold); font-weight:700;">km/h (MẠNH NHẤT)</div>
          </div>
        </div>
      `;
      document.getElementById("btn-jump-best-shot")?.addEventListener("click", () => {
        window.__jumpToShot(bestShot.idx);
      });
    } else {
      bestShotCardEl.innerHTML = `<div style="font-size:11px; color:var(--text-muted);">Chưa ghi nhận cú chạm cầu nào.</div>`;
    }
  }

  // 2. Render Player Scouting Cards
  let cardsHtml = "";
  for (const p of matchData.players) {
    const s = matchData.summary[p.id];
    if (!s) continue;
    const pShots = matchData.shots.filter((sh) => sh.player_id === p.id);
    let maxShotKmh = 0;
    for (const sh of pShots) {
      if (sh.speed_kmh && sh.speed_kmh > maxShotKmh) maxShotKmh = sh.speed_kmh;
    }
    const maxRunKmh = (s.max_speed_mps * 3.6).toFixed(1);
    const styleInfo = evaluatePlayerStyle(s, pShots);
    const fPct = Math.round(s.zones_pct.front || 0);
    const mPct = Math.round(s.zones_pct.mid || 0);
    const bPct = Math.max(0, 100 - fPct - mPct);

    cardsHtml += `
      <div class="scouting-card">
        <div class="scouting-header">
          <div style="display:flex; align-items:center; gap:6px;">
            <span class="color-dot" style="background:${p.color};"></span>
            <span style="color:${p.color}; font-weight:700; font-size:12px;">${p.name}</span>
            <span style="font-size:10px; color:var(--text-muted);">(Đội ${p.team})</span>
          </div>
          <span class="style-badge">${styleInfo.badge}</span>
        </div>
        <div style="font-size:10.5px; color:#cbd5e1;">${styleInfo.desc}</div>
        <div class="scouting-grid">
          <div>
            <div class="stat-cell-val" style="color:var(--cyan);">${pShots.length}</div>
            <div class="stat-cell-lbl">Chạm cầu</div>
          </div>
          <div>
            <div class="stat-cell-val" style="color:var(--gold);">${maxShotKmh ? Math.round(maxShotKmh) : "--"}</div>
            <div class="stat-cell-lbl">Cầu max (km/h)</div>
          </div>
          <div>
            <div class="stat-cell-val">${s.total_distance_m.toFixed(1)}m</div>
            <div class="stat-cell-lbl">Quãng đường</div>
          </div>
          <div>
            <div class="stat-cell-val" style="color:var(--emerald);">${maxRunKmh}</div>
            <div class="stat-cell-lbl">Chạy max (km/h)</div>
          </div>
        </div>
        <div>
          <div style="display:flex; justify-content:space-between; font-size:9.5px; color:var(--text-muted); margin-bottom:3px;">
            <span>Trên lưới: <b style="color:#38bdf8">${fPct}%</b></span>
            <span>Giữa sân: <b style="color:#34d399">${mPct}%</b></span>
            <span>Cuối sân: <b style="color:#facc15">${bPct}%</b></span>
          </div>
          <div class="zone-bar">
            <div style="width:${fPct}%; background:#38bdf8;"></div>
            <div style="width:${mPct}%; background:#34d399;"></div>
            <div style="width:${bPct}%; background:#facc15;"></div>
          </div>
        </div>
      </div>
    `;
  }
  summaryPlayersEl.innerHTML = cardsHtml;

  // 3. Draw Court Heatmap
  const hctx = heatmapCanvas.getContext("2d");
  const hw = heatmapCanvas.width;
  const hh = heatmapCanvas.height;
  hctx.fillStyle = "#091824";
  hctx.fillRect(0, 0, hw, hh);
  hctx.strokeStyle = "#334155";
  hctx.strokeRect(2, 2, hw - 4, hh - 4);
  hctx.strokeStyle = "#22d3ee";
  hctx.beginPath();
  hctx.moveTo(0, hh / 2);
  hctx.lineTo(hw, hh / 2);
  hctx.stroke();

  const cw_m = matchData.court.dims_m.width || 6.1;
  for (const f of matchData.frames) {
    for (const p of matchData.players) {
      const pf = f.players[p.id];
      if (!pf || !pf.foot_court_m) continue;
      const hx = (pf.foot_court_m[0] / cw_m) * hw;
      const hy = (pf.foot_court_m[1] / 13.4) * hh;
      hctx.fillStyle = p.color + "14";
      hctx.beginPath();
      hctx.arc(hx, hy, 9, 0, Math.PI * 2);
      hctx.fill();
    }
  }
  window.__heatmapDrawn = true;

  // 4. Render Detailed Shot List with Shot Type Badges & Top Speed Highlight
  shotsListEl.innerHTML = "";
  matchData.shots.forEach((s, idx) => {
    const pInfo = playerMap[s.player_id] || { name: s.player_id, color: "#e2e8f0" };
    const stKey = inferShotTypeClient(s, idx);
    const stMeta = SHOT_TYPE_VN[stKey] || SHOT_TYPE_VN.Drive;
    const isBest = bestShot && s.idx === bestShot.idx && s.speed_kmh;
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.dataset.shotIdx = String(s.idx);
    btn.style.cssText = `text-align:left; padding:5px 8px; font-size:11px; display:flex; align-items:center; justify-content:space-between; border-color:${isBest ? "rgba(250,204,21,0.55)" : "var(--border)"};`;
    btn.innerHTML = `
      <span>
        <b>#${s.idx}</b> <span style="color:var(--text-muted);">[${(s.frame / fps).toFixed(1)}s]</span>
        <span style="color:${pInfo.color}; font-weight:600;">${pInfo.name}</span>
        <span class="shot-badge" style="color:${stMeta.color}; background:${stMeta.bg};">${stMeta.label}</span>
      </span>
      <span style="font-family:var(--font-num); font-size:13px; font-weight:700; color:${isBest ? "var(--gold)" : "#fff"};">
        ${isBest ? "🔥 " : ""}${s.speed_kmh ? Math.round(s.speed_kmh) : "--"} km/h
      </span>
    `;
    btn.addEventListener("click", () => window.__jumpToShot(s.idx));
    shotsListEl.appendChild(btn);
  });
}

window.__jumpToShot = async (shotIdx) => {
  const shot = matchData.shots.find((s) => s.idx === Number(shotIdx));
  if (!shot) return;
  setCalibOverlayActive(false);
  const targetSec = shot.frame / matchData.video.fps;
  await window.__seekToTime(targetSec);
};

btnDeleteData?.addEventListener("click", async () => {
  window.__dataDeleted = true;
  btnDeleteData.textContent = "Đã xóa!";
  await fetch("/api/match", { method: "DELETE" });
});

// ============================================================================
// UI CONTROLS, STREAMING UPLOAD & LIVE AI PIPELINE POLLING
// ============================================================================
function showDockTab(which) {
  const isSum = which === "summary";
  tabBtnSetup?.classList.toggle("active", !isSum);
  tabBtnSummary?.classList.toggle("active", isSum);
  setupPanel.style.display = isSum ? "none" : "flex";
  summaryPanel.style.display = isSum ? "flex" : "none";
  if (isSum) renderSummaryPanel();
}

tabBtnSetup?.addEventListener("click", () => showDockTab("setup"));
tabBtnSummary?.addEventListener("click", () => showDockTab("summary"));

navHud?.addEventListener("click", () => {
  [navHud, navSetup, navSummary].forEach((b) => b?.classList.remove("active"));
  navHud.classList.add("active");
  studioDock.classList.remove("mobile-open");
  setCalibOverlayActive(false);
});
navSetup?.addEventListener("click", () => {
  [navHud, navSetup, navSummary].forEach((b) => b?.classList.remove("active"));
  navSetup.classList.add("active");
  studioDock.classList.add("mobile-open");
  showDockTab("setup");
  setCalibOverlayActive(true);
});
navSummary?.addEventListener("click", () => {
  [navHud, navSetup, navSummary].forEach((b) => b?.classList.remove("active"));
  navSummary.classList.add("active");
  studioDock.classList.add("mobile-open");
  showDockTab("summary");
  setCalibOverlayActive(false);
});

btnToggleCalib?.addEventListener("click", () => setCalibOverlayActive(!isCalibMode));
btnAutoCalib?.addEventListener("click", () => window.__resetCalibValid());
btnAutoAiSetup?.addEventListener("click", async () => {
  btnAutoAiSetup.disabled = true;
  const origTxt = btnAutoAiSetup.textContent;
  btnAutoAiSetup.textContent = "⏳ Đang dò sân & người...";
  calibStatusEl.textContent = "Đang dùng AI tự động nhận diện 6 điểm vạch sân & 4 người chơi...";
  calibStatusEl.style.color = "#38bdf8";
  try {
    const res = await fetch("/api/auto-setup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: modeSelect.value, useUploadedVideo: hasUploadedVideo }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || "Không tự động nhận diện được");
    if (Array.isArray(data.calib_points) && data.calib_points.length === 6) {
      window.__calibState.points = data.calib_points;
    }
    if (Array.isArray(data.initial_players) && data.initial_players.length > 0) {
      window.__assignedPlayers = data.initial_players.map((p) => ({
        id: p.id,
        name: p.name,
        team: p.team,
        px: p.click_px,
      }));
      renderPlayerEditRows();
    }
    setCalibOverlayActive(true);
    btnProcess.disabled = false;
    calibStatusEl.textContent = `Đã tự động canh 6 điểm sân & ${window.__assignedPlayers.length} người chơi! Bấm "Phân Tích & Phủ HUD" ngay.`;
    calibStatusEl.style.color = "#4ade80";
  } catch (err) {
    window.__resetCalibValid();
    setCalibOverlayActive(true);
    calibStatusEl.textContent = `Đã đặt khung 6 điểm chuẩn (${err.message || err})`;
  } finally {
    btnAutoAiSetup.disabled = false;
    btnAutoAiSetup.textContent = origTxt;
  }
});
btnClearCalib?.addEventListener("click", () => {
  window.__calibState.points = [];
  window.__assignedPlayers = [];
  renderPlayerEditRows();
  setCalibOverlayActive(true);
});

modeSelect.addEventListener("change", () => {
  const targets = getTargetCourtCoords(modeSelect.value);
  window.__calibState.points.forEach((p, idx) => {
    if (targets[idx]) p.court_m = targets[idx];
  });
  renderCalibOverlay();
});

btnPlayPause?.addEventListener("click", () => {
  if (isCalibMode) setCalibOverlayActive(false);
  if (video.paused) {
    video.play().catch(() => {});
    btnPlayPause.textContent = "⏸ Dừng";
  } else {
    video.pause();
    btnPlayPause.textContent = "▶ Phát";
  }
});

function seekRelativeSec(deltaSec) {
  const fps = matchData?.video?.fps || 30;
  const maxDur = video.duration && Number.isFinite(video.duration) ? video.duration : (matchData?.video?.duration_sec || 15);
  const nextSec = Math.max(0, Math.min(maxDur, (video.currentTime || 0) + deltaSec));
  video.currentTime = nextSec;
  const fIdx = Math.round(nextSec * fps);
  if (isCalibMode) renderCalibOverlay();
  else renderFrame(fIdx);
}

btnSeekBack?.addEventListener("click", () => seekRelativeSec(-3));
btnSeekFwd?.addEventListener("click", () => seekRelativeSec(3));

// Smooth Video Scrubbing (prevents rAF / requestVideoFrameCallback from fighting the user's thumb)
timelineSlider?.addEventListener("pointerdown", () => {
  isScrubbing = true;
});
window.addEventListener("pointerup", () => {
  if (isScrubbing) isScrubbing = false;
});
timelineSlider?.addEventListener("input", (e) => {
  isScrubbing = true;
  const fps = matchData?.video?.fps || 30;
  const fIdx = Number(e.target.value);
  video.pause();
  if (btnPlayPause) btnPlayPause.textContent = "▶ Phát";
  video.currentTime = fIdx / fps;
  if (isCalibMode) renderCalibOverlay();
  else renderFrame(fIdx);
});
timelineSlider?.addEventListener("change", (e) => {
  isScrubbing = false;
  const fps = matchData?.video?.fps || 30;
  const fIdx = Number(e.target.value);
  video.currentTime = fIdx / fps;
  if (isCalibMode) renderCalibOverlay();
  else renderFrame(fIdx);
});

// Video Audio Mute & Volume Controls + Web Audio SFX
function syncAudioUI() {
  const isMuted = video.muted || video.volume === 0;
  if (btnMute) btnMute.textContent = isMuted ? "🔇" : "🔊";
  if (audioStateLabel) {
    audioStateLabel.textContent = isMuted ? "Video: Tắt tiếng" : `Âm lượng: ${Math.round(video.volume * 100)}%`;
  }
}

window.__toggleMute = (forceMute) => {
  video.muted = typeof forceMute === "boolean" ? forceMute : !video.muted;
  if (!video.muted && video.volume === 0) {
    video.volume = 0.8;
    if (volumeSlider) volumeSlider.value = "80";
  }
  syncAudioUI();
  return !video.muted;
};

btnMute?.addEventListener("click", () => {
  window.__toggleMute();
  if (!video.muted && chkSfx?.checked) {
    playShuttleHitSfx(180, video.volume);
  }
});

volumeSlider?.addEventListener("input", (e) => {
  const v = Math.max(0, Math.min(1, Number(e.target.value) / 100));
  video.volume = v;
  video.muted = v === 0;
  syncAudioUI();
});

// Match Scoring Controls & Point History
function renderScoreUI() {
  const sc = scoreManager.state;
  if (scoreDisplay) scoreDisplay.textContent = `${sc.teamA} - ${sc.teamB}`;
  if (setDisplay) setDisplay.textContent = `Set: ${sc.setA} - ${sc.setB} · Giao cầu: Đội ${sc.serving}`;
  if (dockScoreA) dockScoreA.textContent = String(sc.teamA);
  if (dockScoreB) dockScoreB.textContent = String(sc.teamB);
  if (scoreHistoryList) {
    scoreHistoryList.innerHTML = "";
    sc.history.slice(-12).forEach((item, idx) => {
      const chip = document.createElement("button");
      chip.className = "btn";
      chip.style.cssText = "padding:2px 6px; font-size:10px;";
      chip.textContent = `${item.team} (${item.scoreA}-${item.scoreB}) @${item.timeSec.toFixed(1)}s`;
      chip.title = "Bấm để tua lại thời điểm ghi điểm này";
      chip.addEventListener("click", () => {
        window.__seekToTime(item.timeSec);
      });
      scoreHistoryList.appendChild(chip);
    });
  }
  if (!isCalibMode) renderFrame(window.__currentFrameIdx || 0);
}

window.__addMatchPoint = (team) => {
  manualScoreOverride = true;
  const fIdx = window.__currentFrameIdx || 0;
  const tSec = video.currentTime || (matchData ? fIdx / matchData.video.fps : 0);
  scoreManager.addPoint(team, fIdx, tSec);
  if (chkSfx?.checked) playScoreChimeSfx(video.muted ? 0.65 : video.volume || 0.8);
  renderScoreUI();
  return scoreManager.state;
};

window.__undoMatchPoint = () => {
  manualScoreOverride = true;
  scoreManager.undo();
  renderScoreUI();
  return scoreManager.state;
};

window.__resetMatchScore = () => {
  manualScoreOverride = false;
  lastAutoRallyId = 0;
  scoreManager.reset();
  renderScoreUI();
  return scoreManager.state;
};

btnScoreA?.addEventListener("click", () => window.__addMatchPoint("A"));
btnScoreB?.addEventListener("click", () => window.__addMatchPoint("B"));
btnDockScoreA?.addEventListener("click", () => window.__addMatchPoint("A"));
btnDockScoreB?.addEventListener("click", () => window.__addMatchPoint("B"));
btnScoreUndo?.addEventListener("click", () => window.__undoMatchPoint());
btnScoreReset?.addEventListener("click", () => window.__resetMatchScore());

// Keyboard shortcuts (Space: play/pause, Left/Right: seek, A/B: score, Z: undo)
window.addEventListener("keydown", (e) => {
  const tag = (e.target?.tagName || "").toUpperCase();
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.code === "Space") {
    e.preventDefault();
    btnPlayPause?.click();
  } else if (e.code === "ArrowLeft") {
    e.preventDefault();
    seekRelativeSec(-1);
  } else if (e.code === "ArrowRight") {
    e.preventDefault();
    seekRelativeSec(1);
  } else if (e.key === "a" || e.key === "A") {
    window.__addMatchPoint("A");
  } else if (e.key === "b" || e.key === "B") {
    window.__addMatchPoint("B");
  } else if (e.key === "z" || e.key === "Z") {
    window.__undoMatchPoint();
  }
});

let speedIdx = 0;
const SPEEDS = [1.0, 0.5, 0.25];
btnSpeed?.addEventListener("click", () => {
  speedIdx = (speedIdx + 1) % SPEEDS.length;
  video.playbackRate = SPEEDS[speedIdx];
  btnSpeed.textContent = `${SPEEDS[speedIdx]}x`;
});

// Instant Local Video Preview + Streaming Upload with Progress
videoFileInput?.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  // 1. Show user's video IMMEDIATELY on the stage so they can see & calibrate right away!
  const localBlobUrl = URL.createObjectURL(file);
  video.pause();
  video.src = localBlobUrl;
  video.currentTime = 0;

  uploadProgTrack.style.display = "block";
  uploadProgFill.style.width = "2%";
  uploadStatusEl.textContent = `Đang tải video lên máy chủ (0%) — ${file.name}`;
  uploadStatusEl.style.color = "#38bdf8";

  const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")) : ".mp4";
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/upload-video?ext=${encodeURIComponent(ext)}`);

  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) {
      const pct = Math.round((ev.loaded / ev.total) * 90);
      uploadProgFill.style.width = `${pct}%`;
      uploadStatusEl.textContent = `Đang tải video (${pct}%) & dò người chơi bằng YOLOv8-Pose...`;
    }
  };

  xhr.onload = () => {
    let data = null;
    try {
      data = JSON.parse(xhr.responseText);
    } catch {
      uploadStatusEl.textContent = "Máy chủ chưa cập nhật API — Hãy tắt và chạy lại `npm run dev`!";
      uploadStatusEl.style.color = "#f43f5e";
      return;
    }
    if (xhr.status !== 200 || !data.ok) {
      uploadStatusEl.textContent = `Lỗi tải video: ${data.error || xhr.status}`;
      uploadStatusEl.style.color = "#f43f5e";
      return;
    }

    uploadProgFill.style.width = "100%";
    if (data.transcoded && data.videoUrl) {
      video.src = data.videoUrl;
      video.currentTime = 0;
    }
    videoMetaDim = { width: data.width, height: data.height };
    fitStageToVideo();

    // Populate detected court corners & YOLOv8-Pose players right onto the video!
    if (Array.isArray(data.calib_points) && data.calib_points.length === 6) {
      window.__calibState.points = data.calib_points;
    } else {
      window.__resetCalibValid();
    }

    if (Array.isArray(data.initial_players) && data.initial_players.length > 0) {
      window.__assignedPlayers = data.initial_players.map((p) => ({
        id: p.id,
        name: p.name,
        team: p.team,
        px: p.click_px,
      }));
      renderPlayerEditRows();
    }

    hasUploadedVideo = true;
    setCalibOverlayActive(true);
    btnProcess.disabled = false;
    uploadStatusEl.textContent = `Đã tự động canh sân (6 điểm) & nhận diện ${window.__assignedPlayers.length} người chơi! Bấm "Phân Tích & Phủ HUD" ngay hoặc tinh chỉnh điểm.`;
    uploadStatusEl.style.color = "#4ade80";
  };

  xhr.onerror = () => {
    uploadStatusEl.textContent = "Lỗi kết nối tới máy chủ local!";
    uploadStatusEl.style.color = "#f43f5e";
  };

  xhr.send(file);
});

// Run AI Pipeline & Poll Progress
btnCancelProcess?.addEventListener("click", async () => {
  btnCancelProcess.disabled = true;
  uploadStatusEl.textContent = "Đang dừng tiến trình AI...";
  try {
    await fetch("/api/cancel-process", { method: "POST" });
  } catch (err) {
    console.warn("Lỗi khi gửi yêu cầu hủy:", err);
  }
});

btnProcess?.addEventListener("click", async () => {
  btnProcess.disabled = true;
  if (btnCancelProcess) {
    btnCancelProcess.disabled = false;
    btnCancelProcess.style.display = "block";
  }
  const origLabel = btnProcess.textContent;
  btnProcess.textContent = "Đang chạy AI (10%)...";
  uploadProgTrack.style.display = "block";
  uploadProgFill.style.width = "10%";
  uploadStatusEl.textContent = "Đang khởi tạo mô hình AI...";
  uploadStatusEl.style.color = "#38bdf8";

  const payload = {
    mode: modeSelect.value,
    max_frames: maxFramesSelect ? Number(maxFramesSelect.value) : 300,
    calib_points: window.__calibState.points,
    initial_players: window.__assignedPlayers.map((p, i) => ({
      id: p.id,
      name: p.name,
      team: p.team,
      color: PLAYER_COLORS[i % PLAYER_COLORS.length],
      click_px: p.px,
    })),
  };

  try {
    const startRes = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const startText = await startRes.text();
    let startJson;
    try {
      startJson = JSON.parse(startText);
    } catch {
      throw new Error("Mất kết nối tới server (Cloudflare 502/504). Hãy kiểm tra Cell 4 trên Kaggle xem còn đang chạy không.");
    }
    if (!startRes.ok) throw new Error(startJson.error || "Không khởi chạy được pipeline");

    // Poll /api/process-status with retry
    let pollFailures = 0;
    while (true) {
      await new Promise((r) => setTimeout(r, 600));
      let st;
      try {
        const stRes = await fetch("/api/process-status");
        const stText = await stRes.text();
        st = JSON.parse(stText);
        pollFailures = 0;
      } catch {
        pollFailures++;
        if (pollFailures > 6) {
          throw new Error("Mất kết nối máy chủ Kaggle (Cloudflare 502/504). Tiến trình trên Kaggle có thể đã bị dừng hoặc hết bộ nhớ.");
        }
        continue;
      }
      uploadProgFill.style.width = `${st.progress || 15}%`;
      btnProcess.textContent = `Đang chạy AI (${st.progress || 15}%)...`;
      if (st.message && (st.status === "PROCESSING" || st.status === "CANCELLED")) {
        uploadStatusEl.textContent = st.message;
      }
      if (st.status === "CANCELLED") {
        uploadStatusEl.textContent = "Đã hủy tiến trình AI.";
        uploadStatusEl.style.color = "#f43f5e";
        btnProcess.textContent = origLabel;
        btnProcess.disabled = false;
        if (btnCancelProcess) btnCancelProcess.style.display = "none";
        break;
      }
      if (st.status === "COMPLETED" && st.match) {
        matchData = st.match;
        window.__matchData = matchData;
        videoMetaDim = { width: matchData.video.width, height: matchData.video.height };
        fitStageToVideo();
        setCalibOverlayActive(false);
        renderSummaryPanel();
        uploadStatusEl.textContent = `Hoàn tất! Đã đồng bộ HUD cho ${matchData.video.total_frames} frames (${matchData.shots.length} cú đánh).`;
        uploadStatusEl.style.color = "#4ade80";
        btnProcess.textContent = origLabel;
        btnProcess.disabled = false;
        if (btnCancelProcess) btnCancelProcess.style.display = "none";
        video.currentTime = 0;
        video.play().catch(() => { });
        btnPlayPause.textContent = "⏸ Dừng";
        break;
      }
      if (st.status === "FAILED") {
        throw new Error(st.error || "Pipeline thất bại");
      }
    }
  } catch (err) {
    uploadStatusEl.textContent = `Lỗi xử lý AI: ${err.message || err}`;
    uploadStatusEl.style.color = "#f43f5e";
    btnProcess.textContent = origLabel;
    btnProcess.disabled = false;
    if (btnCancelProcess) btnCancelProcess.style.display = "none";
  }
});

// ============================================================================
// INITIALIZE
// ============================================================================
async function init() {
  const tabParam = new URLSearchParams(window.location.search).get("tab");
  const res = await fetch("/api/match");
  matchData = await res.json();
  window.__matchData = matchData;
  videoMetaDim = { width: matchData.video.width, height: matchData.video.height };

  fitStageToVideo();
  startSyncLoop();

  if (tabParam === "setup") {
    studioDock.classList.add("mobile-open");
    showDockTab("setup");
    setCalibOverlayActive(true);
  } else if (tabParam === "summary") {
    studioDock.classList.add("mobile-open");
    renderFrame(90);
    showDockTab("summary");
    setCalibOverlayActive(false);
  } else {
    renderFrame(0);
  }
}

init();
