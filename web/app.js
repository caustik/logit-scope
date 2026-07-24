const elements = Object.fromEntries([
  "connection", "profile", "diversity", "diversityValue", "candidateCap", "minimumRelativeProbability", "seed",
  "protectControlTokens", "plot", "scopeCaption", "shapedLegend", "poolMass",
  "jsShift", "effectiveChoices", "peak", "selectedTokenLabel", "selectedToken", "transcript", "message", "clear", "stop",
  "send", "status", "error"
].map(id => [id, document.getElementById(id)]));

let snapshot = null;
let controlsInitialized = false;
let lastTranscript = null;
let settingTimer = 0;
let pendingSettings = null;
let settingsRequestInFlight = false;
let connected = false;
let staleSamplingStep = null;
let displayedSamplingView = null;
const maximumDisplayPointCount = 64;

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
    diversity: Number(elements.diversity.value) / 100,
    candidateCap: Number(elements.candidateCap.value),
    minimumRelativeProbability: Number(elements.minimumRelativeProbability.value),
    seed: Number(elements.seed.value),
    protectControlTokens: elements.protectControlTokens.checked
  };
}

function updateControlLabels() {
  const shapingDisabled = elements.profile.value === "none";
  const diversity = Number(elements.diversity.value) / 100;
  elements.diversity.disabled = shapingDisabled;
  elements.diversityValue.value = shapingDisabled ? "Bypassed" : `${elements.diversity.value}% · ${diversityMode(diversity)}`;
}

function diversityMode(diversity) {
  if (diversity === 0) return "deterministic";
  if (diversity < 1) return "sharpen";
  if (diversity > 1) return `${diversity.toFixed(2)}× choices`;
  return "raw";
}

function queueSettings() {
  staleSamplingStep = snapshot?.samplingStep ?? 0;
  updateControlLabels();
  pendingSettings = currentSettings();
  window.clearTimeout(settingTimer);
  settingTimer = window.setTimeout(flushSettings, 35);
  drawPlot();
}

async function flushSettings() {
  window.clearTimeout(settingTimer);
  settingTimer = 0;
  if (settingsRequestInFlight || !pendingSettings) return;

  const settings = pendingSettings;
  pendingSettings = null;
  settingsRequestInFlight = true;
  try {
    await request("/api/settings", { method: "POST", body: JSON.stringify(settings) });
    applySnapshot(await request("/api/snapshot"));
    elements.error.textContent = "";
  } catch (error) {
    elements.error.textContent = error.message;
  } finally {
    settingsRequestInFlight = false;
    if (pendingSettings) settingTimer = window.setTimeout(flushSettings, 35);
  }
}

function initializeControls(settings) {
  elements.profile.value = settings.profile;
  elements.candidateCap.value = String(settings.candidateCap);
  const requestedFloor = Number(settings.minimumRelativeProbability);
  const nearestFloorOption = Array.from(elements.minimumRelativeProbability.options).reduce((nearest, option) =>
    Math.abs(Number(option.value) - requestedFloor) < Math.abs(Number(nearest.value) - requestedFloor) ? option : nearest);
  elements.minimumRelativeProbability.value = nearestFloorOption.value;
  elements.diversity.value = Math.round(settings.diversity * 100);
  elements.seed.value = settings.seed;
  elements.protectControlTokens.checked = settings.protectControlTokens;
  updateControlLabels();
  controlsInitialized = true;
}

function displayRanks(candidateCount, maximumCount = maximumDisplayPointCount) {
  const count = Math.max(1, candidateCount);
  if (count === 1) return [0];
  const displayCount = Math.min(maximumCount, count);
  if (count <= displayCount) return Array.from({ length: count }, (_, rank) => rank);

  const ranks = [];
  for (let displayIndex = 0; displayIndex < displayCount; ++displayIndex) {
    const fraction = displayIndex / (displayCount - 1);
    const logarithmicRank = Math.round(Math.expm1(fraction * Math.log(count)));
    const minimumRank = displayIndex === 0 ? 0 : ranks[displayIndex - 1] + 1;
    const maximumRank = count - (displayCount - displayIndex);
    ranks.push(Math.max(minimumRank, Math.min(maximumRank, logarithmicRank)));
  }
  return ranks;
}

function sampledProbabilities(values, ranks) {
  return values.map((value, index) => ({ rank: Number(ranks[index] ?? index), value }));
}

function hasCurrentSamplingData(settings, next = snapshot) {
  return Boolean(next?.samplingStep) && next.samplingStep !== staleSamplingStep &&
    next.candidateCount > 0 && next.candidateCount <= settings.candidateCap;
}

function settingsMatch(left, right) {
  return left?.profile === right.profile &&
    Math.abs(left.diversity - right.diversity) < 0.0001 &&
    left.candidateCap === right.candidateCap &&
    Math.abs(left.minimumRelativeProbability - right.minimumRelativeProbability) < 0.000001;
}

function samplingSettingsMatch(left, right) {
  return settingsMatch(left, right) &&
    left.seed === right.seed &&
    left.protectControlTokens === right.protectControlTokens;
}

function samplingView(settings, next = snapshot) {
  if (hasCurrentSamplingData(settings, next) && samplingSettingsMatch(next.samplingSettings, settings)) {
    return {
      data: next,
      preview: false,
      settings: next.settings,
      representativeSampling: next.representativeSampling,
      selectedToken: next.selectedToken
    };
  }
  if (next?.preview && settingsMatch(next.settings, settings) &&
      next.preview.candidateCount > 0 && next.preview.candidateCount <= settings.candidateCap) {
    return { data: next.preview, preview: true, settings: next.settings };
  }
  return null;
}

function stableSamplingView(settings, next = snapshot) {
  const currentView = samplingView(settings, next);
  if (currentView) displayedSamplingView = currentView;
  return { current: Boolean(currentView), view: currentView || displayedSamplingView };
}

function probabilityExponentRange(probabilities, fitToData) {
  if (!fitToData || !probabilities.length) return { maximum: 0, minimum: -6 };

  const maximum = Math.min(0, Math.ceil(Math.log10(Math.max(...probabilities))));
  const dataMinimum = Math.floor(Math.log10(Math.min(...probabilities)));
  return {
    maximum,
    minimum: Math.max(maximum - 6, Math.min(maximum - 2, dataMinimum))
  };
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
  const stableView = stableSamplingView(settings);
  const view = stableView.view;
  const plotSettings = view?.settings || settings;
  const shapingActive = plotSettings.profile !== "none" && plotSettings.diversity !== 1;
  elements.shapedLegend.hidden = !shapingActive;
  const candidateCount = Math.max(1, view?.data.candidateCount || plotSettings.candidateCap);
  const rankDomainCount = Math.max(2, plotSettings.candidateCap);
  const rankDenominator = Math.log(rankDomainCount);
  const probabilityRanks = view?.data.probabilityRanks || [];
  const raw = sampledProbabilities(view?.data.rawProbabilities || [], probabilityRanks);
  const shaped = sampledProbabilities(view?.data.shapedProbabilities || [], probabilityRanks);
  const plottedProbabilities = [...raw, ...shaped].map(point => point.value).filter(value => value > 0);
  const exponentRange = probabilityExponentRange(plottedProbabilities, Boolean(view?.preview));
  const maximumExponent = exponentRange.maximum;
  const minimumExponent = exponentRange.minimum;
  const x = rank => margin.left + (Math.log1p(rank) / rankDenominator) * plotWidth;
  const y = probability => {
    const exponent = Math.log10(Math.max(Math.pow(10, minimumExponent), probability));
    return margin.top + ((maximumExponent - exponent) / (maximumExponent - minimumExponent)) * plotHeight;
  };

  context.font = "10px ui-sans-serif, system-ui";
  context.textBaseline = "middle";
  for (let exponent = maximumExponent; exponent >= minimumExponent; --exponent) {
    const probability = Math.pow(10, exponent);
    const axisY = y(probability);
    context.strokeStyle = exponent === maximumExponent || exponent === minimumExponent ? "#45515e" : "#27313a";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(margin.left, axisY);
    context.lineTo(w - margin.right, axisY);
    context.stroke();
    context.fillStyle = "#7f8d9a";
    const percentage = probability * 100;
    const decimalPlaces = percentage >= 1 ? 0 : Math.min(6, Math.max(1, -Math.floor(Math.log10(percentage))));
    const label = `${percentage.toFixed(decimalPlaces)}%`;
    context.fillText(label, 7, axisY);
  }

  const axisRanks = displayRanks(rankDomainCount, 5);
  for (const rank of axisRanks) {
    const axisX = x(rank);
    context.strokeStyle = rank === 0 || rank === rankDomainCount - 1 ? "#45515e" : "#27313a";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(axisX, margin.top);
    context.lineTo(axisX, h - margin.bottom);
    context.stroke();
  }

  const drawCurve = (points, color, width, alpha = 1) => {
    const visiblePoints = points.filter(point => point.rank >= 0 && point.rank < candidateCount);
    if (visiblePoints.length === 1) {
      context.beginPath();
      context.arc(x(visiblePoints[0].rank), y(visiblePoints[0].value), 2.5, 0, Math.PI * 2);
      context.fillStyle = color;
      context.globalAlpha = alpha;
      context.fill();
      context.globalAlpha = 1;
      return;
    }
    if (visiblePoints.length < 2) return;
    context.strokeStyle = color;
    context.globalAlpha = alpha;
    context.lineWidth = width;
    context.lineJoin = "round";
    context.beginPath();
    visiblePoints.forEach((point, index) => {
      const px = x(point.rank);
      const py = y(point.value);
      if (index === 0) context.moveTo(px, py); else context.lineTo(px, py);
    });
    context.stroke();
    context.globalAlpha = 1;
  };

  drawCurve(raw, "#43d0c1", 1.8);
  if (shapingActive) drawCurve(shaped, "#ff9b42", 2.0);

  context.fillStyle = "#7f8d9a";
  context.textBaseline = "alphabetic";
  axisRanks.forEach((rank, index) => {
    context.textAlign = index === 0 ? "left" : index === axisRanks.length - 1 ? "right" : "center";
    const label = index === 0 ? "Top 1" : index === axisRanks.length - 1 ? `Cap ${rankDomainCount}` : `Top ${rank + 1}`;
    context.fillText(label, x(rank), h - 7);
  });
  context.textAlign = "left";

  const diversityStatus = settings.profile === "none" ? "bypassed" :
    `${Math.round(settings.diversity * 100)}% · ${diversityMode(settings.diversity)}`;
  if (!stableView.current && view) {
    elements.scopeCaption.textContent = `Updating preview · ${diversityStatus} · showing previous curve`;
  } else if (view?.preview) {
    elements.scopeCaption.textContent =
      `Illustrative preview · ${diversityStatus} · ${candidateCount} retained · fitted probability range`;
  } else if (view) {
    const tokenStatus = view.representativeSampling ? "most uncertain token" : "selected token";
    elements.scopeCaption.textContent =
      `Actual token probabilities · ${diversityStatus} · ${candidateCount} retained · ${tokenStatus} ${JSON.stringify(view.selectedToken || "")}`;
  } else {
    elements.scopeCaption.textContent = `Preparing preview · ${diversityStatus}`;
  }
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

function percentage(value, digits = null) {
  if (!Number.isFinite(value)) return "—";
  const percent = value * 100;
  const decimalPlaces = digits ?? (percent < 0.1 ? 3 : percent < 1 ? 2 : 1);
  return `${percent.toFixed(decimalPlaces)}%`;
}

function effectiveChoiceCount(entropy) {
  if (!Number.isFinite(entropy)) return "—";
  const count = Math.exp(entropy);
  return count < 10 ? count.toFixed(1) : count < 100 ? count.toFixed(0) : Math.round(count).toLocaleString();
}

function applySnapshot(next) {
  snapshot = next;
  if (!controlsInitialized) initializeControls(next.settings);
  if (staleSamplingStep !== null && next.samplingStep > 0 && next.samplingStep !== staleSamplingStep) staleSamplingStep = null;
  connected = true;
  elements.connection.textContent = "Local";
  elements.connection.className = "connection online";
  elements.status.textContent = next.generating ? `${next.status} · sampling step ${next.samplingStep}` : next.status;
  const view = stableSamplingView(currentSettings(), next).view;
  const data = view?.data;
  elements.poolMass.textContent = data && !view.preview ? percentage(data.poolProbabilityMass) : "—";
  elements.jsShift.textContent = data ? `${data.jensenShannonDivergence.toFixed(4)} nats` : "—";
  elements.effectiveChoices.textContent = data ? `${effectiveChoiceCount(data.rawEntropy)} → ${effectiveChoiceCount(data.shapedEntropy)}` : "—";
  elements.peak.textContent = data ? `${percentage(data.rawPeakProbability)} → ${percentage(data.shapedPeakProbability)}` : "—";
  elements.selectedToken.textContent = view?.preview ? "Illustrative rank curve" : data ? view.selectedToken || "—" : "—";
  elements.selectedTokenLabel.textContent = view?.preview ? "Preview" :
    view?.representativeSampling ? "Most uncertain token" : "Selected token";
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

for (const element of [elements.profile, elements.seed, elements.protectControlTokens]) element.addEventListener("change", queueSettings);
elements.diversity.addEventListener("input", queueSettings);
elements.candidateCap.addEventListener("change", queueSettings);
elements.minimumRelativeProbability.addEventListener("change", queueSettings);

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
