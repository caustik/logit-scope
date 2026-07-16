const elements = Object.fromEntries([
  "connection", "profile", "blend", "blendValue", "concentration", "concentrationValue",
  "candidateCount", "seed", "protectControlTokens", "plot", "scopeCaption", "poolMass",
  "jsShift", "entropy", "peak", "selectedToken", "transcript", "message", "clear", "stop",
  "send", "status", "error"
].map(id => [id, document.getElementById(id)]));

let snapshot = null;
let controlsInitialized = false;
let lastTranscript = null;
let settingTimer = 0;
let connected = false;

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function currentSettings() {
  return {
    profile: elements.profile.value,
    blend: Number(elements.blend.value),
    concentration: Number(elements.concentration.value),
    candidateCount: Number(elements.candidateCount.value),
    seed: Number(elements.seed.value),
    protectControlTokens: elements.protectControlTokens.checked
  };
}

function updateControlLabels() {
  elements.blendValue.value = `${Math.round(Number(elements.blend.value) * 100)}%`;
  elements.concentrationValue.value = Number(elements.concentration.value).toFixed(2);
}

function queueSettings() {
  updateControlLabels();
  window.clearTimeout(settingTimer);
  settingTimer = window.setTimeout(async () => {
    try {
      await request("/api/settings", { method: "POST", body: JSON.stringify(currentSettings()) });
      elements.error.textContent = "";
      drawPlot();
    } catch (error) {
      elements.error.textContent = error.message;
    }
  }, 35);
}

function initializeControls(settings) {
  elements.profile.value = settings.profile;
  elements.blend.value = settings.blend;
  elements.concentration.value = settings.concentration;
  elements.candidateCount.value = String(settings.candidateCount);
  elements.seed.value = settings.seed;
  elements.protectControlTokens.checked = settings.protectControlTokens;
  updateControlLabels();
  controlsInitialized = true;
}

function targetProbabilities(settings) {
  const count = Math.max(2, settings.candidateCount);
  const concentration = Math.max(0.05, Math.min(4, settings.concentration));
  const values = new Array(count);
  let total = 0;
  for (let rank = 0; rank < count; ++rank) {
    let value = 1;
    if (settings.profile === "exponential") value = Math.exp(-concentration * rank);
    else if (settings.profile === "power") value = Math.pow(rank + 1, -concentration);
    else if (settings.profile === "half-normal") value = Math.exp(-0.5 * concentration * concentration * rank * rank);
    values[rank] = Math.max(value, Number.MIN_VALUE);
    total += values[rank];
  }
  return values.slice(0, 64).map(value => value / total);
}

function drawPlot() {
  const canvas = elements.plot;
  const rectangle = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rectangle.width * ratio));
  const height = Math.max(1, Math.round(rectangle.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const w = rectangle.width;
  const h = rectangle.height;
  const margin = { left: 54, right: 14, top: 14, bottom: 27 };
  const plotWidth = Math.max(1, w - margin.left - margin.right);
  const plotHeight = Math.max(1, h - margin.top - margin.bottom);
  context.clearRect(0, 0, w, h);

  const settings = currentSettings();
  const target = targetProbabilities(settings);
  const raw = snapshot?.rawProbabilities || [];
  const shaped = snapshot?.shapedProbabilities || [];
  const pointCount = Math.max(2, target.length, raw.length, shaped.length);
  const x = rank => margin.left + (rank / Math.max(1, pointCount - 1)) * plotWidth;
  const y = probability => {
    const log = Math.log10(Math.max(1e-6, Math.min(1, probability)));
    return margin.top + (-log / 6) * plotHeight;
  };

  context.font = "10px ui-sans-serif, system-ui";
  context.textBaseline = "middle";
  for (let exponent = 0; exponent >= -6; --exponent) {
    const probability = Math.pow(10, exponent);
    const axisY = y(probability);
    context.strokeStyle = exponent === 0 || exponent === -6 ? "#45515e" : "#27313a";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(margin.left, axisY);
    context.lineTo(w - margin.right, axisY);
    context.stroke();
    context.fillStyle = "#7f8d9a";
    const label = exponent === 0 ? "100%" : exponent === -1 ? "10%" : exponent === -2 ? "1%" : `${Math.pow(10, exponent + 2)}%`;
    context.fillText(label, 7, axisY);
  }

  const drawCurve = (values, color, width, alpha = 1) => {
    if (!values || values.length < 2) return;
    context.strokeStyle = color;
    context.globalAlpha = alpha;
    context.lineWidth = width;
    context.lineJoin = "round";
    context.beginPath();
    values.forEach((value, rank) => {
      const px = x(rank);
      const py = y(value);
      if (rank === 0) context.moveTo(px, py); else context.lineTo(px, py);
    });
    context.stroke();
    context.globalAlpha = 1;
  };

  drawCurve(target, "#b779ff", 2.2, 0.92);
  drawCurve(raw, "#43d0c1", 1.8);
  drawCurve(shaped, "#ff9b42", 2.0);

  context.fillStyle = "#7f8d9a";
  context.textBaseline = "alphabetic";
  context.fillText("Rank 1", margin.left, h - 7);
  context.textAlign = "right";
  context.fillText(`Rank ${pointCount}`, w - margin.right, h - 7);
  context.textAlign = "left";
}

function updateTranscript(text) {
  const wasNearBottom = elements.transcript.scrollHeight - elements.transcript.scrollTop - elements.transcript.clientHeight < 45;
  if (lastTranscript !== null && text.startsWith(lastTranscript)) {
    elements.transcript.append(document.createTextNode(text.slice(lastTranscript.length)));
  } else {
    elements.transcript.textContent = text || "Start a conversation.";
  }
  lastTranscript = text;
  if (wasNearBottom) elements.transcript.scrollTop = elements.transcript.scrollHeight;
}

function percentage(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function applySnapshot(next) {
  snapshot = next;
  if (!controlsInitialized) initializeControls(next.settings);
  connected = true;
  elements.connection.textContent = "Local";
  elements.connection.className = "connection online";
  elements.status.textContent = next.generating ? `${next.status} · sampling step ${next.samplingStep}` : next.status;
  elements.scopeCaption.textContent = `Top ${next.candidateCount || next.settings.candidateCount} candidates${next.samplingStep ? ` · sampling step ${next.samplingStep}` : " · persistent target"}`;
  elements.poolMass.textContent = next.samplingStep ? percentage(next.poolProbabilityMass) : "—";
  elements.jsShift.textContent = next.samplingStep ? `${next.jensenShannonDivergence.toFixed(4)} nats` : "—";
  elements.entropy.textContent = next.samplingStep ? `${next.rawEntropy.toFixed(2)} → ${next.shapedEntropy.toFixed(2)} nats` : "—";
  elements.peak.textContent = next.samplingStep ? `${percentage(next.rawPeakProbability)} → ${percentage(next.shapedPeakProbability)}` : "—";
  elements.selectedToken.textContent = next.selectedToken || "—";
  updateTranscript(next.transcript || "");

  const canSend = next.modelLoaded && !next.generating;
  elements.send.disabled = !canSend;
  elements.message.disabled = !canSend;
  elements.stop.disabled = !next.generating;
  elements.clear.disabled = !next.modelLoaded;
  drawPlot();
}

async function poll() {
  try {
    applySnapshot(await request("/api/snapshot"));
    elements.error.textContent = "";
  } catch (error) {
    if (connected) elements.error.textContent = error.message;
    connected = false;
    elements.connection.textContent = "Offline";
    elements.connection.className = "connection offline";
  } finally {
    window.setTimeout(poll, snapshot?.generating ? 75 : 250);
  }
}

async function sendMessage() {
  const message = elements.message.value.trim();
  if (!message) return;
  try {
    elements.send.disabled = true;
    await request("/api/message", { method: "POST", body: JSON.stringify({ message }) });
    elements.message.value = "";
    elements.error.textContent = "";
  } catch (error) {
    elements.error.textContent = error.message;
  }
}

for (const element of [elements.profile, elements.blend, elements.concentration, elements.candidateCount, elements.seed, elements.protectControlTokens]) {
  element.addEventListener(element.type === "range" ? "input" : "change", queueSettings);
}

elements.send.addEventListener("click", sendMessage);
elements.stop.addEventListener("click", () => request("/api/stop", { method: "POST", body: "{}" }).catch(error => elements.error.textContent = error.message));
elements.clear.addEventListener("click", () => request("/api/clear", { method: "POST", body: "{}" }).catch(error => elements.error.textContent = error.message));
elements.message.addEventListener("keydown", event => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    sendMessage();
  }
});
window.addEventListener("resize", drawPlot);

updateControlLabels();
drawPlot();
poll();
