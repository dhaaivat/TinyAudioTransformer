// app.js: TinyAudioTransformer keyword-spotting demo. Everything runs in the browser:
//   microphone (any rate) → resample to 16 kHz → pick the 1 s around the word → ONNX graph
//   (Kaldi fbank + transformer, exported by export_onnx.py) → probabilities over 35 words.
// No audio leaves the device.
"use strict";

const ORT_VERSION = "1.22.0";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const SR = 16000;            // the model's sample rate
const CLIP = 16000;          // and its input length: 1 s
const RECORD_SECONDS = 2.0;  // a bit longer than 1 s, so you don't have to start speaking instantly
const SILENCE_RMS = 0.003;   // quieter than this (about −50 dBFS) counts as "didn't hear anything"
const UNSURE = 0.5;          // below this top probability, say that it's a guess
// Browser "voice processing" changes the sound in ways training never saw, so it's switched off.
const MIC = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };

const $ = (id) => document.getElementById(id);
let session, classes, lastClip = null, busy = false, META = {};

// ---------- audio ----------

async function recordMic(seconds, progressBar) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: MIC });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  // ScriptProcessorNode is deprecated but works everywhere without a separate worklet file
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(proc);
  proc.connect(ctx.destination);  // must be connected to run; it outputs silence
  const t0 = performance.now();
  await new Promise((resolve) => {
    const tick = () => {
      const f = Math.min((performance.now() - t0) / (seconds * 1000), 1);
      progressBar.style.width = `${f * 100}%`;
      f < 1 ? requestAnimationFrame(tick) : resolve();
    };
    tick();
  });
  proc.disconnect();
  source.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const rate = ctx.sampleRate;
  await ctx.close();
  progressBar.style.width = "0";
  return resample(concat(chunks), rate);
}

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let i = 0;
  for (const c of chunks) { out.set(c, i); i += c.length; }
  return out;
}

async function resample(samples, fromRate) {
  // Mics usually run at 44.1 or 48 kHz; let the browser's own resampler bring it to 16 kHz
  if (fromRate === SR) return samples;
  const off = new OfflineAudioContext(1, Math.round(samples.length * SR / fromRate), SR);
  const buf = off.createBuffer(1, samples.length, fromRate);
  buf.copyToChannel(samples, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

async function decodeFile(file) {
  // decodeAudioData on a 16 kHz context decodes *and* resamples to 16 kHz
  const audio = await new OfflineAudioContext(1, 1, SR).decodeAudioData(await file.arrayBuffer());
  return audio.getChannelData(0);
}

function centredSecond(x) {
  // Speech Commands clips are 1 s with the word roughly in the middle (training's ±100 ms shift covers
  // small offsets). Find the word as the energy centroid of loud 10 ms frames, and centre 1 s on it.
  const out = new Float32Array(CLIP);
  if (x.length <= CLIP) { out.set(x); return out; }
  const hop = 160, energy = [];
  for (let s = 0; s + hop <= x.length; s += hop) {
    let e = 0;
    for (let i = s; i < s + hop; i++) e += x[i] * x[i];
    energy.push(e);
  }
  const peak = Math.max(...energy);
  let w = 0, c = 0;
  energy.forEach((e, i) => { if (e >= peak * 0.05) { w += e; c += e * (i + 0.5) * hop; } });  // within 13 dB of the peak
  const centre = w > 0 ? c / w : x.length / 2;
  // Outside the recording counts as silence (zeros, like training's padding), so a word spoken right at
  // the start or end still ends up centred instead of the window being clamped to the recording.
  const start = Math.round(centre - CLIP / 2);
  for (let i = Math.max(0, -start); i < CLIP && start + i < x.length; i++) out[i] = x[start + i];
  return out;
}

function rms(x) {
  let s = 0;
  for (const v of x) s += v * v;
  return Math.sqrt(s / x.length);
}

// ---------- model ----------

async function classify(clip) {
  const out = await session.run({ waveform: new ort.Tensor("float32", clip, [1, CLIP]) });
  const logits = Array.from(out.logits.data);
  const m = Math.max(...logits);
  const exp = logits.map((l) => Math.exp(l - m));
  const z = exp.reduce((a, b) => a + b, 0);
  const probs = exp.map((e) => e / z);
  const ranked = probs.map((p, i) => ({ word: classes[i], p })).sort((a, b) => b.p - a.p);
  return { ranked, fbank: out.fbank.data };
}

// ---------- drawing ----------

const STOPS = [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]];  // magma-like

function colour(t) {
  const x = Math.min(Math.max(t, 0), 1) * (STOPS.length - 1), i = Math.min(Math.floor(x), STOPS.length - 2), f = x - i;
  return STOPS[i].map((a, k) => Math.round(a + f * (STOPS[i + 1][k] - a)));
}

function drawSpectrogram(canvas, fbank, frames = 128, bins = 128, realFrames = 98) {
  // fbank: [frames, bins] row-major. Colour scale: the loudest real value down to 12 nats (≈ 52 dB) below it.
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(frames, bins);
  let vmax = -Infinity;
  for (let t = 0; t < realFrames; t++) for (let b = 0; b < bins; b++) vmax = Math.max(vmax, fbank[t * bins + b]);
  const vmin = vmax - 12;
  for (let t = 0; t < frames; t++) {
    for (let b = 0; b < bins; b++) {
      const [r, g, bl] = colour((fbank[t * bins + b] - vmin) / (vmax - vmin));
      const o = ((bins - 1 - b) * frames + t) * 4;  // low pitch at the bottom
      img.data.set([r, g, bl, 255], o);
    }
  }
  const tmp = Object.assign(document.createElement("canvas"), { width: frames, height: bins });
  tmp.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  const x = (realFrames / frames) * canvas.width;
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = "#ffffff";
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
}

function showResult(res, level) {
  const [top] = res.ranked;
  $("result").hidden = false;
  $("word").textContent = top.word;
  $("conf").textContent = `${(top.p * 100).toFixed(1)}% confident`;
  $("note").textContent = top.p < UNSURE
    ? "Low confidence: this is a guess. Was it one of the 35 words below?"
    : `Input level ${(20 * Math.log10(level)).toFixed(0)} dBFS.`;
  $("top5").innerHTML = res.ranked.slice(0, 5).map((r) =>
    `<li><span>${r.word}</span><span class="track"><span class="fill" style="width:${(r.p * 100).toFixed(1)}%"></span></span>` +
    `<span class="pct">${(r.p * 100).toFixed(1)}%</span></li>`).join("");
  drawSpectrogram($("spec"), res.fbank);
}

function playClip(clip) {
  const ctx = new AudioContext();
  const buf = ctx.createBuffer(1, clip.length, SR);
  buf.copyToChannel(clip, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.onended = () => ctx.close();
  src.start();
}

// ---------- "Try it" ----------

async function analyse(samples) {
  const clip = centredSecond(samples), level = rms(clip);
  lastClip = clip;
  if (level < SILENCE_RMS) {
    $("status").textContent = "I didn't hear anything. Try speaking louder or closer to the microphone.";
    return null;
  }
  const res = await classify(clip);
  showResult(res, level);
  return { res, level };
}

async function guarded(statusEl, fn) {
  if (busy) return;
  busy = true;
  setEnabled(false);
  try {
    await fn();
  } catch (err) {
    statusEl.textContent = err.name === "NotAllowedError"
      ? "Microphone access was blocked. Allow it in the browser's address bar and try again."
      : `Something went wrong: ${err.message}`;
    console.error(err);
  } finally {
    busy = false;
    setEnabled(true);
  }
}

$("record").addEventListener("click", () => guarded($("status"), async () => {
  $("status").textContent = "Listening… say one word.";
  const samples = await recordMic(RECORD_SECONDS, $("progress"));
  if (await analyse(samples)) $("status").textContent = "Done. Record again any time.";
}));

$("upload").addEventListener("change", (e) => guarded($("status"), async () => {
  const file = e.target.files[0];
  if (!file) return;
  $("status").textContent = `Analysing ${file.name}…`;
  if (await analyse(await decodeFile(file))) $("status").textContent = `Analysed ${file.name}.`;
  e.target.value = "";
}));

$("play").addEventListener("click", () => lastClip && playClip(lastClip));

// ---------- "Test it": the real-user evaluation ----------

const STORE_KEY = "tat-test-trials-v1";
let queue = [], trials = loadTrials();

function loadTrials() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; } catch { return []; }
}

function saveTrials() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(trials)); } catch { /* storage unavailable: keep in memory */ }
}

function shuffled(a) {
  const b = a.slice();
  for (let i = b.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [b[i], b[j]] = [b[j], b[i]];
  }
  return b;
}

function renderTest() {
  $("target").textContent = queue.length ? queue[0] : "—";
  const n = trials.length, correct = trials.filter((t) => t.correct).length;
  $("t-score").textContent = n ? `${correct} / ${n} correct (${(100 * correct / n).toFixed(1)}%)` +
    (queue.length ? ` · ${queue.length} words left in this round` : "") : "";
  $("t-table").hidden = !n;
  $("t-table").querySelector("tbody").innerHTML = trials.slice().reverse().map((t) =>
    `<tr><td>${t.target}</td><td>${t.predicted}</td><td>${(t.confidence * 100).toFixed(0)}%</td>` +
    `<td class="${t.correct ? "ok" : "miss"}">${t.correct ? "✓" : "✗"}</td></tr>`).join("");
  setEnabled(!busy);
}

$("t-start").addEventListener("click", () => {
  queue = shuffled(classes);
  $("t-feedback").textContent = "Press Record, then say the word shown.";
  renderTest();
});

$("t-skip").addEventListener("click", () => {
  queue.shift();
  $("t-feedback").textContent = "";
  renderTest();
});

$("t-record").addEventListener("click", () => guarded($("t-feedback"), async () => {
  const target = queue[0];
  $("t-feedback").textContent = `Listening… say "${target}".`;
  const samples = await recordMic(RECORD_SECONDS, $("t-progress"));
  const clip = centredSecond(samples), level = rms(clip);
  if (level < SILENCE_RMS) {
    $("t-feedback").textContent = "I didn't hear anything. Try again (this attempt wasn't counted).";
    return;
  }
  const { ranked } = await classify(clip);
  const correct = ranked[0].word === target;
  trials.push({
    target, predicted: ranked[0].word, confidence: ranked[0].p, correct,
    target_rank: ranked.findIndex((r) => r.word === target) + 1,
    top5: ranked.slice(0, 5).map((r) => [r.word, +r.p.toFixed(4)]),
    level_dbfs: +(20 * Math.log10(level)).toFixed(1), time: new Date().toISOString(),
  });
  saveTrials();
  queue.shift();
  $("t-feedback").textContent = correct
    ? `✓ Heard "${ranked[0].word}" (${(ranked[0].p * 100).toFixed(0)}%).`
    : `✗ Heard "${ranked[0].word}" (${(ranked[0].p * 100).toFixed(0)}%) instead of "${target}".`;
  if (!queue.length) $("t-feedback").textContent += " Round finished. Thank you! Download the results below.";
  renderTest();
}));

$("t-download").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({
    model: META.checkpoint, model_val_acc: META.val_acc, page: location.href, user_agent: navigator.userAgent,
    mic_settings: MIC, downloaded: new Date().toISOString(), trials,
  }, null, 2)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: `tat-real-user-test-${Date.now()}.json`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ---------- tabs, state, start-up ----------

function setEnabled(on) {
  const ready = on && !!session;
  $("record").disabled = !ready;
  $("upload").disabled = !ready;
  $("upload-label").classList.toggle("disabled", !ready);
  $("t-start").disabled = !ready;
  $("t-record").disabled = !ready || !queue.length;
  $("t-skip").disabled = !ready || !queue.length;
  $("t-download").disabled = !trials.length;
}

for (const name of ["try", "test"]) {
  $(`tab-${name}`).addEventListener("click", () => {
    for (const other of ["try", "test"]) {
      $(`tab-${other}`).setAttribute("aria-selected", String(other === name));
      $(`panel-${other}`).hidden = other !== name;
    }
  });
}

async function init() {
  try {
    META = await (await fetch("model/meta.json")).json();
    classes = META.classes;
    $("words").innerHTML = classes.map((w) => `<span class="chip">${w}</span>`).join("");
    session = await ort.InferenceSession.create("model/tat.onnx", { executionProviders: ["wasm"] });
    // Warm-up run, so the first real click isn't slowed by one-time setup
    await session.run({ waveform: new ort.Tensor("float32", new Float32Array(CLIP), [1, CLIP]) });
    $("status").textContent = "Ready. Press Record and say one of the words below.";
  } catch (err) {
    $("status").textContent = `Couldn't load the model: ${err.message}`;
    console.error(err);
  }
  renderTest();
}

init();
