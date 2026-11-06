// ponytail: native Web Audio API synthesizer + BWF 21-point match scoring state (zero external deps)

let audioCtx = null;

function getAudioContext() {
  if (!audioCtx && typeof window !== "undefined") {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (Ctor) audioCtx = new Ctor();
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

/**
 * Synthesizes a realistic badminton racket-shuttlecock impact sound ("thwack")
 * using native Web Audio API (short bandpass noise burst + pitch-drop body).
 */
export function playShuttleHitSfx(speedKmh = 180, volume = 0.8) {
  const ctx = getAudioContext();
  if (!ctx || volume <= 0) return false;
  const now = ctx.currentTime;
  const intensity = Math.min(1.3, Math.max(0.5, (speedKmh || 160) / 220));

  // 1. Resonant string-bed impact tone (fast pitch drop 520Hz -> 130Hz)
  const osc = ctx.createOscillator();
  const oscGain = ctx.createGain();
  osc.type = "triangle";
  osc.frequency.setValueAtTime(480 * intensity, now);
  osc.frequency.exponentialRampToValueAtTime(125, now + 0.045);

  oscGain.gain.setValueAtTime(0.35 * volume * intensity, now);
  oscGain.gain.exponentialRampToValueAtTime(0.001, now + 0.055);

  osc.connect(oscGain);
  oscGain.connect(ctx.destination);
  osc.start(now);
  osc.stop(now + 0.06);

  // 2. Crisp cork/feather snap (high-passed white noise burst)
  const bufSize = Math.floor(ctx.sampleRate * 0.04);
  const noiseBuf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
  const data = noiseBuf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) {
    data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSize * 0.28));
  }
  const noise = ctx.createBufferSource();
  noise.buffer = noiseBuf;

  const filter = ctx.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.setValueAtTime(2400 * intensity, now);
  filter.Q.setValueAtTime(1.8, now);

  const noiseGain = ctx.createGain();
  noiseGain.gain.setValueAtTime(0.45 * volume * intensity, now);
  noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);

  noise.connect(filter);
  filter.connect(noiseGain);
  noiseGain.connect(ctx.destination);
  noise.start(now);
  return true;
}

/**
 * Synthesizes a clean broadcast scoreboard chime when a point is scored.
 */
export function playScoreChimeSfx(volume = 0.8) {
  const ctx = getAudioContext();
  if (!ctx || volume <= 0) return false;
  const now = ctx.currentTime;
  const freqs = [587.33, 880]; // D5 -> A5
  freqs.forEach((f, idx) => {
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = "sine";
    const t0 = now + idx * 0.07;
    osc.frequency.setValueAtTime(f, t0);
    g.gain.setValueAtTime(0.2 * volume, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.18);
    osc.connect(g);
    g.connect(ctx.destination);
    osc.start(t0);
    osc.stop(t0 + 0.19);
  });
  return true;
}

/**
 * BWF 21-point Badminton Match Scoring Manager
 */
export function createScoreManager() {
  const state = {
    teamA: 0,
    teamB: 0,
    setA: 0,
    setB: 0,
    serving: "A",
    history: [],
  };

  function isGameWon(a, b) {
    if (a >= 30 || b >= 30) return a !== b;
    if (a >= 21 || b >= 21) return Math.abs(a - b) >= 2;
    return false;
  }

  return {
    state,
    addPoint(team, frameIdx = 0, timeSec = 0) {
      const snapshot = {
        prevA: state.teamA,
        prevB: state.teamB,
        prevSetA: state.setA,
        prevSetB: state.setB,
        prevServing: state.serving,
        team,
        frameIdx,
        timeSec,
      };
      if (team === "A") state.teamA += 1;
      else if (team === "B") state.teamB += 1;
      state.serving = team;

      const won = isGameWon(state.teamA, state.teamB);
      if (won) {
        if (state.teamA > state.teamB) state.setA += 1;
        else state.setB += 1;
      }

      snapshot.scoreA = state.teamA;
      snapshot.scoreB = state.teamB;
      snapshot.setWon = won;
      state.history.push(snapshot);

      if (won) {
        state.teamA = 0;
        state.teamB = 0;
      }
      return state;
    },
    undo() {
      const last = state.history.pop();
      if (!last) return state;
      state.teamA = last.prevA;
      state.teamB = last.prevB;
      state.setA = last.prevSetA;
      state.setB = last.prevSetB;
      state.serving = last.prevServing;
      return state;
    },
    reset() {
      state.teamA = 0;
      state.teamB = 0;
      state.setA = 0;
      state.setB = 0;
      state.serving = "A";
      state.history = [];
      return state;
    },
  };
}
