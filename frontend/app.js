const MAX_RECORD_SECONDS = 15;

const $ = (id) => document.getElementById(id);
const sections = { working: $("working"), error: $("error"), result: $("result") };

function show(name) {
  for (const [key, el] of Object.entries(sections)) el.classList.toggle("hidden", key !== name);
}

function setBusy(busy) {
  document.querySelectorAll("button, #upload-input").forEach((el) => {
    if (el.id !== "record-btn" || !recorder) el.disabled = busy;
  });
}

// ---------- API ----------

async function compose(request) {
  show("working");
  setBusy(true);
  try {
    const response = await request();
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.message || "Something went wrong. Please try again.");
    renderResult(body);
  } catch (err) {
    $("error-message").textContent = err.message;
    show("error");
  } finally {
    setBusy(false);
  }
}

const composeFromSample = (id) => compose(() => fetch(`/api/symphony/sample/${id}`, { method: "POST" }));

function composeFromBlob(blob, filename) {
  const form = new FormData();
  form.append("file", blob, filename);
  return compose(() => fetch("/api/symphony", { method: "POST", body: form }));
}

// ---------- Result card ----------

function renderResult(r) {
  $("species-name").textContent = r.species.common_name;
  $("species-sci").textContent = r.species.scientific_name;
  $("species-confidence").textContent = `${Math.round(r.species.confidence * 100)}% match`;

  const facts = $("facts");
  facts.replaceChildren();
  const rows = r.facts
    ? [["Habitat", r.facts.habitat], ["Song", r.facts.song], ["Did you know?", r.facts.fun_fact]]
    : [["Facts", "No offline facts stored for this species yet."]];
  for (const [label, text] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = text;
    facts.append(dt, dd);
  }

  $("melody-title").textContent = r.melody.title;
  $("player").src = r.audio_url;
  $("download").href = r.audio_url;
  $("download").download = r.download_name;
  $("composer-note").textContent = r.composer_note;

  const [lo, hi] = r.bird_pitch_range_hz;
  const shift = r.octave_shift;
  $("pitch-note").textContent =
    `The bird sang mostly between ${Math.round(lo)} and ${Math.round(hi)} Hz` +
    (shift ? `, so the melody is moved ${Math.abs(shift)} octave${Math.abs(shift) > 1 ? "s" : ""} ${shift < 0 ? "down" : "up"} to sit in a comfortable range.` : ".");

  const notes = $("notes");
  notes.replaceChildren(
    ...r.melody.notes.map((n) => {
      const chip = document.createElement("span");
      chip.className = n.pitch === "rest" ? "note rest" : "note";
      chip.textContent = n.pitch === "rest" ? "·" : n.pitch;
      chip.title = `${n.pitch}, ${n.beats} beat${n.beats === 1 ? "" : "s"}`;
      return chip;
    }),
  );
  show("result");
}

// ---------- Samples ----------

async function loadSamples() {
  const samples = await (await fetch("/api/samples")).json();
  const list = $("sample-list");
  const credits = [];
  for (const s of samples) {
    const li = document.createElement("li");
    li.innerHTML = `
      <div><div class="title"></div><div class="muted small desc"></div></div>
      <div class="sample-actions">
        <audio controls preload="none"></audio>
        <button class="btn secondary">Compose</button>
      </div>`;
    li.querySelector(".title").textContent = s.title;
    li.querySelector(".desc").textContent = s.description;
    li.querySelector("audio").src = `/api/samples/${s.id}/audio`;
    li.querySelector("button").addEventListener("click", () => composeFromSample(s.id));
    list.append(li);
    credits.push(`${s.title}: ${s.credit}, ${s.license}`);
  }
  $("credits").textContent = `Sample credits — ${credits.join(" · ")}. Bird ID by BirdNET (Cornell Lab of Ornithology & Chemnitz University of Technology).`;
}

async function loadStatus() {
  const pill = $("composer-status");
  try {
    const { composer, composer_model } = await (await fetch("/api/health")).json();
    pill.textContent = {
      ready: `AI composer ready (${composer_model})`,
      model_missing: `AI model "${composer_model}" not installed — using direct transcription`,
      offline: "AI composer offline — using direct transcription",
    }[composer];
  } catch {
    pill.textContent = "Server unreachable";
  }
}

// ---------- Recording ----------
// Records raw samples and builds a WAV in the browser, so the server needs no extra decoders.

let recorder = null;

async function startRecording() {
  // Browsers' speech filters treat birdsong as noise; switch them all off.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("recorder-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "recorder-processor");
  const chunks = [];
  node.port.onmessage = (e) => chunks.push(e.data);
  source.connect(node);

  const started = Date.now();
  const timer = setInterval(() => {
    const seconds = (Date.now() - started) / 1000;
    $("record-timer").textContent = `${seconds.toFixed(1)} s`;
    if (seconds >= MAX_RECORD_SECONDS) stopRecording();
  }, 100);

  recorder = { stream, ctx, source, node, chunks, timer };
  $("record-btn").textContent = "■ Stop";
  $("record-btn").classList.add("recording");
}

async function stopRecording() {
  if (!recorder) return;
  const { stream, ctx, source, node, chunks, timer } = recorder;
  recorder = null;
  clearInterval(timer);
  source.disconnect();
  node.port.onmessage = null;
  stream.getTracks().forEach((t) => t.stop());
  const sampleRate = ctx.sampleRate;
  await ctx.close();

  $("record-btn").textContent = "● Record";
  $("record-btn").classList.remove("recording");
  $("record-timer").textContent = "";
  await composeFromBlob(encodeWav(chunks, sampleRate), "recording.wav");
}

function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => [...text].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));

  writeText(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true); // format chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per sample
  view.setUint16(34, 16, true); // bits per sample
  writeText(36, "data");
  view.setUint32(40, length * 2, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (const sample of chunk) {
      const s = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([view], { type: "audio/wav" });
}

$("record-btn").addEventListener("click", async () => {
  if (recorder) return stopRecording();
  try {
    await startRecording();
  } catch {
    $("error-message").textContent = "Couldn't access the microphone. Please allow microphone access, or try a sample recording.";
    show("error");
  }
});

$("upload-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) composeFromBlob(file, file.name);
  e.target.value = "";
});

loadStatus();
loadSamples();
