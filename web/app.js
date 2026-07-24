const elements = Object.fromEntries([
  "connection", "profile", "diversity", "diversityValue", "candidateCap", "minimumRelativeProbability", "seed",
  "protectControlTokens", "plot", "scopeCaption", "shapedLegend", "poolMass",
  "jsShift", "effectiveChoices", "peak", "selectedTokenLabel", "selectedToken", "transcript", "message", "clear", "stop",
  "send", "status", "error", "evalNameA", "evalProfileA", "evalDiversityA", "evalCapA", "evalFloorA", "evalGuardA",
  "evalUseCurrentA", "evalNameB", "evalProfileB", "evalDiversityB", "evalCapB", "evalFloorB", "evalGuardB",
  "evalUseCurrentB", "evalPrompts", "evalRepeats", "evalSeedStart", "evalStart", "evalCancel", "evalRunStatus",
  "evalProgress", "evalJudge", "evalTrialLabel", "evalPromptDisplay", "evalNext", "evalLeftResponse", "evalRightResponse",
  "evalLeftMetrics", "evalRightMetrics", "evalRubric", "evalTaskLeft", "evalTaskRight", "evalCoherenceLeft",
  "evalCoherenceRight", "evalStyleLeft", "evalStyleRight", "evalNotes", "evalSubmitJudgment", "evalReveal",
  "evalSummary", "evalExperiment", "evalSummaryBody", "evalExportJson", "evalExportCsv", "evalDelete"
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
let evaluationDefaultsInitialized = false;
let evaluationRunning = false;
let evaluationCancelRequested = false;
let currentEvaluationTrialId = null;
const maximumDisplayPointCount = 64;
const evaluationStorageKey = "logit-scope-evaluations-v1";
const evaluationActiveKey = "logit-scope-active-evaluation-v1";
let evaluationStore = loadEvaluationStore();
let activeEvaluationId = loadActiveEvaluationId();

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
  if (!evaluationDefaultsInitialized) {
    setEvaluationConfig("A", { ...settings, profile: "none", diversity: 1 });
    setEvaluationConfig("B", settings);
    elements.evalNameA.value = "Raw baseline";
    elements.evalNameB.value = `${profileDisplayName(settings.profile)} ${Math.round(settings.diversity * 100)}%`;
    evaluationDefaultsInitialized = true;
  }
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
  if (lastTranscript && text.startsWith(lastTranscript)) {
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
  updateTranscript(evaluationRunning ? "Blind comparison in progress. Responses stay concealed until each pair is ready." : next.transcript || "");

  const canSend = next.modelLoaded && !next.generating && !evaluationRunning;
  elements.send.disabled = !canSend;
  elements.message.disabled = !canSend;
  elements.stop.disabled = !next.generating;
  elements.clear.disabled = !next.modelLoaded;
  elements.evalStart.disabled = evaluationRunning || !next.modelLoaded || next.generating;
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

function loadEvaluationStore() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(evaluationStorageKey));
    if (stored?.version === 1 && Array.isArray(stored.experiments)) return stored;
  } catch {
    // A malformed or unavailable local store should not block the application.
  }
  return { version: 1, experiments: [] };
}

function loadActiveEvaluationId() {
  try {
    return window.localStorage.getItem(evaluationActiveKey);
  } catch {
    return null;
  }
}

function saveEvaluationStore() {
  evaluationStore.experiments = evaluationStore.experiments.slice(-20);
  try {
    window.localStorage.setItem(evaluationStorageKey, JSON.stringify(evaluationStore));
    if (activeEvaluationId) window.localStorage.setItem(evaluationActiveKey, activeEvaluationId);
  } catch (error) {
    elements.error.textContent = `Evaluation results could not be saved: ${error.message}`;
  }
}

function profileDisplayName(profile) {
  return {
    none: "None",
    exponential: "Exponential",
    soliton: "Soliton",
    power: "Power",
    "half-normal": "Half-normal"
  }[profile] || profile;
}

function evaluationField(side, name) {
  return elements[`eval${name}${side}`];
}

function setEvaluationConfig(side, settings) {
  evaluationField(side, "Profile").value = settings.profile;
  evaluationField(side, "Diversity").value = Math.round(settings.diversity * 100);
  evaluationField(side, "Cap").value = String(settings.candidateCap);
  const floorSelect = evaluationField(side, "Floor");
  const nearestFloor = Array.from(floorSelect.options).reduce((nearest, option) =>
    Math.abs(Number(option.value) - settings.minimumRelativeProbability) <
    Math.abs(Number(nearest.value) - settings.minimumRelativeProbability) ? option : nearest);
  floorSelect.value = nearestFloor.value;
  evaluationField(side, "Guard").checked = settings.protectControlTokens;
  updateEvaluationConfigControls(side);
}

function evaluationSettings(side) {
  return {
    profile: evaluationField(side, "Profile").value,
    diversity: Math.max(0, Math.min(2, Number(evaluationField(side, "Diversity").value) / 100)),
    candidateCap: Number(evaluationField(side, "Cap").value),
    minimumRelativeProbability: Number(evaluationField(side, "Floor").value),
    protectControlTokens: evaluationField(side, "Guard").checked
  };
}

function updateEvaluationConfigControls(side) {
  const bypassed = evaluationField(side, "Profile").value === "none";
  evaluationField(side, "Diversity").disabled = evaluationRunning || bypassed;
}

function useCurrentControlsForEvaluation(side) {
  const settings = currentSettings();
  setEvaluationConfig(side, settings);
  evaluationField(side, "Name").value =
    settings.profile === "none" ? "Raw baseline" : `${profileDisplayName(settings.profile)} ${Math.round(settings.diversity * 100)}%`;
}

function activeEvaluation() {
  return evaluationStore.experiments.find(experiment => experiment.id === activeEvaluationId) || null;
}

function evaluationExperimentLabel(experiment) {
  const date = new Date(experiment.startedAt).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
  const judged = experiment.trials.filter(trial => trial.judgment).length;
  return `${date} · ${experiment.nameA} vs ${experiment.nameB} · ${judged}/${experiment.trials.length} judged`;
}

function renderEvaluationSelector() {
  elements.evalExperiment.replaceChildren();
  if (!evaluationStore.experiments.length) {
    const option = document.createElement("option");
    option.textContent = "No experiments";
    option.value = "";
    elements.evalExperiment.append(option);
    elements.evalExperiment.disabled = true;
    return;
  }

  if (!activeEvaluation()) activeEvaluationId = evaluationStore.experiments.at(-1).id;
  for (const experiment of [...evaluationStore.experiments].reverse()) {
    const option = document.createElement("option");
    option.value = experiment.id;
    option.textContent = evaluationExperimentLabel(experiment);
    elements.evalExperiment.append(option);
  }
  elements.evalExperiment.value = activeEvaluationId;
  elements.evalExperiment.disabled = false;
}

function parseEvaluationPrompts() {
  return elements.evalPrompts.value.split(/\r?\n/).map(prompt => prompt.trim()).filter(Boolean);
}

function randomBoolean() {
  const value = new Uint32Array(1);
  window.crypto.getRandomValues(value);
  return Boolean(value[0] & 1);
}

function createEvaluationId() {
  const random = new Uint32Array(1);
  window.crypto.getRandomValues(random);
  return `${Date.now().toString(36)}-${random[0].toString(36)}`;
}

function setEvaluationRunning(running) {
  evaluationRunning = running;
  const setupControls = [
    "evalNameA", "evalProfileA", "evalDiversityA", "evalCapA", "evalFloorA", "evalGuardA", "evalUseCurrentA",
    "evalNameB", "evalProfileB", "evalDiversityB", "evalCapB", "evalFloorB", "evalGuardB", "evalUseCurrentB",
    "evalPrompts", "evalRepeats", "evalSeedStart"
  ];
  const primaryControls = [
    "profile", "diversity", "candidateCap", "minimumRelativeProbability", "seed", "protectControlTokens", "message",
    "clear", "send"
  ];
  for (const id of [...setupControls, ...primaryControls]) elements[id].disabled = running;
  elements.evalStart.disabled = running || !snapshot?.modelLoaded || snapshot?.generating;
  elements.evalCancel.disabled = !running;
  elements.evalDelete.disabled = running || !activeEvaluation();
  if (!running) {
    updateControlLabels();
    updateEvaluationConfigControls("A");
    updateEvaluationConfigControls("B");
    if (snapshot) applySnapshot(snapshot);
  }
}

function evaluationDelay(milliseconds) {
  return new Promise(resolve => window.setTimeout(resolve, milliseconds));
}

async function generateEvaluationResponse(prompt, settings, seed) {
  const accepted = await request("/api/evaluation", {
    method: "POST",
    body: JSON.stringify({ prompt, settings: { ...settings, seed } })
  });
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    if (evaluationCancelRequested) throw new Error("Comparison stopped");
    const result = await request("/api/evaluation");
    if (result.id === accepted.id && result.ready) return result;
    await evaluationDelay(85);
  }
  throw new Error("Evaluation response timed out");
}

function responseMetrics(response, tokenCount) {
  const words = response.toLocaleLowerCase().match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) || [];
  const trigrams = [];
  for (let index = 0; index + 2 < words.length; ++index) trigrams.push(words.slice(index, index + 3).join(" "));
  const uniqueWords = new Set(words);
  const uniqueTrigrams = new Set(trigrams);
  return {
    tokenCount,
    characterCount: response.length,
    wordCount: words.length,
    uniqueWordRatio: words.length ? uniqueWords.size / words.length : 0,
    repeatedTrigramRatio: trigrams.length ? 1 - uniqueTrigrams.size / trigrams.length : 0
  };
}

async function startEvaluation() {
  if (evaluationRunning) return;
  const prompts = parseEvaluationPrompts();
  const repeats = Math.max(1, Math.min(10, Number(elements.evalRepeats.value) || 1));
  const seedStart = Math.max(0, Math.min(4294967295, Number(elements.evalSeedStart.value) || 0));
  const nameA = elements.evalNameA.value.trim() || "Configuration A";
  const nameB = elements.evalNameB.value.trim() || "Configuration B";
  const settingsA = evaluationSettings("A");
  const settingsB = evaluationSettings("B");
  if (!snapshot?.modelLoaded || snapshot.generating) {
    elements.error.textContent = "The model must be ready before starting a comparison.";
    return;
  }
  if (!prompts.length) {
    elements.error.textContent = "Add at least one evaluation prompt.";
    return;
  }
  if (nameA === nameB) {
    elements.error.textContent = "Give the two configurations distinct names.";
    return;
  }
  if (JSON.stringify(settingsA) === JSON.stringify(settingsB)) {
    elements.error.textContent = "The two configurations are identical.";
    return;
  }

  const experiment = {
    id: createEvaluationId(),
    startedAt: new Date().toISOString(),
    nameA,
    nameB,
    settingsA,
    settingsB,
    prompts,
    repeats,
    seedStart,
    trials: []
  };
  evaluationStore.experiments.push(experiment);
  activeEvaluationId = experiment.id;
  saveEvaluationStore();
  renderEvaluationSelector();
  renderEvaluationSummary();

  evaluationCancelRequested = false;
  setEvaluationRunning(true);
  elements.error.textContent = "";
  const pairCount = prompts.length * repeats;
  let pairIndex = 0;

  try {
    for (let repeat = 0; repeat < repeats; ++repeat) {
      for (const prompt of prompts) {
        if (evaluationCancelRequested) throw new Error("Comparison stopped");
        const seed = (seedStart + pairIndex) % 4294967296;
        const generationOrder = randomBoolean() ? ["A", "B"] : ["B", "A"];
        const responses = {};
        elements.evalProgress.textContent = `${pairIndex + 1} / ${pairCount} pairs`;

        for (let generationIndex = 0; generationIndex < generationOrder.length; ++generationIndex) {
          const key = generationOrder[generationIndex];
          elements.evalRunStatus.textContent =
            `Generating concealed response ${generationIndex + 1} of 2 for pair ${pairIndex + 1}…`;
          responses[key] = await generateEvaluationResponse(prompt, key === "A" ? settingsA : settingsB, seed);
        }

        const trial = {
          id: `${experiment.id}-${pairIndex}`,
          prompt,
          seed,
          generationOrder,
          leftIsA: randomBoolean(),
          responseA: responses.A.response,
          responseB: responses.B.response,
          statusA: responses.A.status,
          statusB: responses.B.status,
          metricsA: responseMetrics(responses.A.response, responses.A.tokenCount),
          metricsB: responseMetrics(responses.B.response, responses.B.tokenCount),
          generatedAt: new Date().toISOString(),
          judgment: null
        };
        experiment.trials.push(trial);
        ++pairIndex;
        saveEvaluationStore();
        renderEvaluationSelector();
        renderEvaluationSummary();
        if (!currentEvaluationTrialId) showEvaluationTrial(trial);
      }
    }
    elements.evalRunStatus.textContent = `Generated ${pairIndex} blinded ${pairIndex === 1 ? "pair" : "pairs"}.`;
    elements.evalProgress.textContent = "Judge every pair before interpreting the aggregate.";
  } catch (error) {
    elements.evalRunStatus.textContent = evaluationCancelRequested ? `Stopped after ${pairIndex} complete pairs.` : "Comparison failed.";
    elements.evalProgress.textContent = "";
    if (!evaluationCancelRequested) elements.error.textContent = error.message;
  } finally {
    setEvaluationRunning(false);
    renderEvaluationSelector();
    renderEvaluationSummary();
    if (!currentEvaluationTrialId) showNextUnjudgedTrial();
  }
}

function stopEvaluation() {
  if (!evaluationRunning) return;
  evaluationCancelRequested = true;
  elements.evalRunStatus.textContent = "Stopping after the current token…";
  request("/api/stop", { method: "POST", body: "{}" }).catch(error => elements.error.textContent = error.message);
}

function evaluationMetricText(metrics) {
  return `${metrics.tokenCount} tokens · ${metrics.wordCount} words · ${(metrics.uniqueWordRatio * 100).toFixed(0)}% unique words · ` +
    `${(metrics.repeatedTrigramRatio * 100).toFixed(1)}% repeated trigrams`;
}

function resetEvaluationRubric() {
  for (const element of [
    elements.evalTaskLeft, elements.evalTaskRight, elements.evalCoherenceLeft, elements.evalCoherenceRight,
    elements.evalStyleLeft, elements.evalStyleRight
  ]) element.value = "";
  for (const preference of document.querySelectorAll('input[name="evalPreference"]')) preference.checked = false;
  elements.evalNotes.value = "";
}

function revealEvaluationTrial(experiment, trial) {
  const leftKey = trial.leftIsA ? "A" : "B";
  const rightKey = trial.leftIsA ? "B" : "A";
  const leftName = leftKey === "A" ? experiment.nameA : experiment.nameB;
  const rightName = rightKey === "A" ? experiment.nameA : experiment.nameB;
  const leftMetrics = leftKey === "A" ? trial.metricsA : trial.metricsB;
  const rightMetrics = rightKey === "A" ? trial.metricsA : trial.metricsB;
  elements.evalLeftMetrics.textContent = evaluationMetricText(leftMetrics);
  elements.evalRightMetrics.textContent = evaluationMetricText(rightMetrics);
  elements.evalReveal.textContent =
    `Revealed: left was ${leftName}; right was ${rightName}. Both used seed ${trial.seed} in isolated one-turn conversations.`;
  elements.evalReveal.hidden = false;
  elements.evalRubric.hidden = true;
}

function showEvaluationTrial(trial) {
  const experiment = activeEvaluation();
  if (!experiment || !trial) {
    elements.evalJudge.hidden = true;
    currentEvaluationTrialId = null;
    return;
  }
  currentEvaluationTrialId = trial.id;
  const trialIndex = experiment.trials.findIndex(candidate => candidate.id === trial.id);
  elements.evalJudge.hidden = false;
  elements.evalTrialLabel.textContent = `Blind trial ${trialIndex + 1} of ${experiment.trials.length}`;
  elements.evalPromptDisplay.textContent = trial.prompt;
  elements.evalLeftResponse.textContent = trial.leftIsA ? trial.responseA : trial.responseB;
  elements.evalRightResponse.textContent = trial.leftIsA ? trial.responseB : trial.responseA;
  elements.evalNext.disabled = !experiment.trials.some(candidate => !candidate.judgment && candidate.id !== trial.id);

  if (trial.judgment) {
    revealEvaluationTrial(experiment, trial);
  } else {
    resetEvaluationRubric();
    elements.evalLeftMetrics.textContent = "Diagnostics reveal after judgment.";
    elements.evalRightMetrics.textContent = "Diagnostics reveal after judgment.";
    elements.evalReveal.hidden = true;
    elements.evalRubric.hidden = false;
  }
}

function showNextUnjudgedTrial() {
  const experiment = activeEvaluation();
  if (!experiment) {
    showEvaluationTrial(null);
    return;
  }
  const next = experiment.trials.find(trial => !trial.judgment && trial.id !== currentEvaluationTrialId) ||
    experiment.trials.find(trial => !trial.judgment);
  if (next) showEvaluationTrial(next);
  else if (!currentEvaluationTrialId && experiment.trials.length) showEvaluationTrial(experiment.trials.at(-1));
}

function submitEvaluationJudgment() {
  const experiment = activeEvaluation();
  const trial = experiment?.trials.find(candidate => candidate.id === currentEvaluationTrialId);
  if (!experiment || !trial || trial.judgment) return;

  const preference = document.querySelector('input[name="evalPreference"]:checked')?.value;
  const ratingElements = [
    elements.evalTaskLeft, elements.evalTaskRight, elements.evalCoherenceLeft, elements.evalCoherenceRight,
    elements.evalStyleLeft, elements.evalStyleRight
  ];
  if (!preference || ratingElements.some(element => !element.value)) {
    elements.error.textContent = "Rate both responses on all three criteria and choose an overall preference.";
    return;
  }

  const leftRatings = {
    taskFit: Number(elements.evalTaskLeft.value),
    coherence: Number(elements.evalCoherenceLeft.value),
    style: Number(elements.evalStyleLeft.value)
  };
  const rightRatings = {
    taskFit: Number(elements.evalTaskRight.value),
    coherence: Number(elements.evalCoherenceRight.value),
    style: Number(elements.evalStyleRight.value)
  };
  let preferred = "tie";
  if (preference !== "tie") {
    const preferredLeft = preference === "left";
    preferred = preferredLeft === trial.leftIsA ? "A" : "B";
  }
  trial.judgment = {
    preferred,
    ratingsA: trial.leftIsA ? leftRatings : rightRatings,
    ratingsB: trial.leftIsA ? rightRatings : leftRatings,
    note: elements.evalNotes.value.trim(),
    judgedAt: new Date().toISOString()
  };
  elements.error.textContent = "";
  saveEvaluationStore();
  revealEvaluationTrial(experiment, trial);
  renderEvaluationSelector();
  renderEvaluationSummary();
  elements.evalNext.disabled = !experiment.trials.some(candidate => !candidate.judgment);
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function exactSignTest(wins, losses) {
  const count = wins + losses;
  if (!count) return null;
  const tail = Math.min(wins, losses);
  let term = Math.pow(0.5, count);
  let cumulative = term;
  for (let index = 1; index <= tail; ++index) {
    term *= (count - index + 1) / index;
    cumulative += term;
  }
  return Math.min(1, 2 * cumulative);
}

function evaluationAggregate(experiment, key) {
  const otherKey = key === "A" ? "B" : "A";
  const judged = experiment.trials.filter(trial => trial.judgment);
  const wins = judged.filter(trial => trial.judgment.preferred === key).length;
  const losses = judged.filter(trial => trial.judgment.preferred === otherKey).length;
  const ties = judged.filter(trial => trial.judgment.preferred === "tie").length;
  const ratings = judged.map(trial => trial.judgment[`ratings${key}`]);
  const metrics = experiment.trials.map(trial => trial[`metrics${key}`]);
  return {
    name: experiment[`name${key}`],
    wins,
    ties,
    losses,
    preference: judged.length ? (wins + ties * 0.5) / judged.length : null,
    taskFit: average(ratings.map(rating => rating.taskFit)),
    coherence: average(ratings.map(rating => rating.coherence)),
    style: average(ratings.map(rating => rating.style)),
    words: average(metrics.map(metric => metric.wordCount)),
    repetition: average(metrics.map(metric => metric.repeatedTrigramRatio))
  };
}

function appendEvaluationCell(row, text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  row.append(cell);
}

function renderEvaluationSummary() {
  const experiment = activeEvaluation();
  elements.evalSummaryBody.replaceChildren();
  const hasExperiment = Boolean(experiment);
  elements.evalExportJson.disabled = !hasExperiment;
  elements.evalExportCsv.disabled = !hasExperiment;
  elements.evalDelete.disabled = evaluationRunning || !hasExperiment;
  if (!experiment) {
    elements.evalSummary.textContent = "No completed judgments yet.";
    return;
  }

  const judged = experiment.trials.filter(trial => trial.judgment);
  const aggregateA = evaluationAggregate(experiment, "A");
  const aggregateB = evaluationAggregate(experiment, "B");
  for (const aggregate of [aggregateA, aggregateB]) {
    const row = document.createElement("tr");
    appendEvaluationCell(row, aggregate.name);
    appendEvaluationCell(row, aggregate.preference === null ? "—" : `${(aggregate.preference * 100).toFixed(1)}%`);
    appendEvaluationCell(row, `${aggregate.wins} / ${aggregate.ties} / ${aggregate.losses}`);
    appendEvaluationCell(row, aggregate.taskFit === null ? "—" : aggregate.taskFit.toFixed(2));
    appendEvaluationCell(row, aggregate.coherence === null ? "—" : aggregate.coherence.toFixed(2));
    appendEvaluationCell(row, aggregate.style === null ? "—" : aggregate.style.toFixed(2));
    appendEvaluationCell(row, aggregate.words === null ? "—" : aggregate.words.toFixed(1));
    appendEvaluationCell(row, aggregate.repetition === null ? "—" : `${(aggregate.repetition * 100).toFixed(1)}%`);
    elements.evalSummaryBody.append(row);
  }

  if (!judged.length) {
    elements.evalSummary.textContent =
      `${experiment.trials.length} generated ${experiment.trials.length === 1 ? "pair" : "pairs"}; judgments are still blind and pending.`;
    return;
  }
  const winsA = judged.filter(trial => trial.judgment.preferred === "A").length;
  const winsB = judged.filter(trial => trial.judgment.preferred === "B").length;
  const ties = judged.length - winsA - winsB;
  const pValue = exactSignTest(winsA, winsB);
  const significance = pValue === null ? "no non-tied decisions yet" :
    `two-sided exact sign-test p=${pValue < 0.001 ? "<0.001" : pValue.toFixed(3)} from ${winsA + winsB} non-tied decisions`;
  elements.evalSummary.textContent =
    `${judged.length}/${experiment.trials.length} judged · ${experiment.nameA} ${winsA} wins, ${experiment.nameB} ${winsB}, ${ties} ties · ${significance}.`;
}

function downloadEvaluation(filename, type, contents) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportEvaluationJson() {
  const experiment = activeEvaluation();
  if (!experiment) return;
  downloadEvaluation(`logit-scope-evaluation-${experiment.id}.json`, "application/json", JSON.stringify(experiment, null, 2));
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : typeof value === "string" ? value : JSON.stringify(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function exportEvaluationCsv() {
  const experiment = activeEvaluation();
  if (!experiment) return;
  const headings = [
    "prompt", "seed", "generation_order", "left_is_a", "preferred", "name_a", "name_b", "settings_a", "settings_b",
    "response_a", "response_b", "status_a", "status_b", "tokens_a", "tokens_b", "task_fit_a", "task_fit_b",
    "coherence_a", "coherence_b", "style_a", "style_b", "words_a", "words_b", "unique_words_a", "unique_words_b",
    "repeated_trigrams_a", "repeated_trigrams_b", "note"
  ];
  const rows = experiment.trials.map(trial => [
    trial.prompt, trial.seed, trial.generationOrder, trial.leftIsA, trial.judgment?.preferred, experiment.nameA,
    experiment.nameB, experiment.settingsA, experiment.settingsB, trial.responseA, trial.responseB, trial.statusA,
    trial.statusB, trial.metricsA.tokenCount, trial.metricsB.tokenCount, trial.judgment?.ratingsA.taskFit,
    trial.judgment?.ratingsB.taskFit, trial.judgment?.ratingsA.coherence, trial.judgment?.ratingsB.coherence,
    trial.judgment?.ratingsA.style, trial.judgment?.ratingsB.style, trial.metricsA.wordCount, trial.metricsB.wordCount,
    trial.metricsA.uniqueWordRatio, trial.metricsB.uniqueWordRatio, trial.metricsA.repeatedTrigramRatio,
    trial.metricsB.repeatedTrigramRatio, trial.judgment?.note
  ]);
  const csv = [headings, ...rows].map(row => row.map(csvCell).join(",")).join("\r\n");
  downloadEvaluation(`logit-scope-evaluation-${experiment.id}.csv`, "text/csv;charset=utf-8", csv);
}

function deleteActiveEvaluation() {
  const experiment = activeEvaluation();
  if (!experiment || evaluationRunning) return;
  if (!window.confirm(`Delete the saved experiment “${experiment.nameA} vs ${experiment.nameB}”?`)) return;
  evaluationStore.experiments = evaluationStore.experiments.filter(candidate => candidate.id !== experiment.id);
  activeEvaluationId = evaluationStore.experiments.at(-1)?.id || null;
  currentEvaluationTrialId = null;
  saveEvaluationStore();
  renderEvaluationSelector();
  renderEvaluationSummary();
  showNextUnjudgedTrial();
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
for (const side of ["A", "B"]) {
  evaluationField(side, "Profile").addEventListener("change", () => updateEvaluationConfigControls(side));
}
elements.evalUseCurrentA.addEventListener("click", () => useCurrentControlsForEvaluation("A"));
elements.evalUseCurrentB.addEventListener("click", () => useCurrentControlsForEvaluation("B"));
elements.evalStart.addEventListener("click", startEvaluation);
elements.evalCancel.addEventListener("click", stopEvaluation);
elements.evalSubmitJudgment.addEventListener("click", submitEvaluationJudgment);
elements.evalNext.addEventListener("click", showNextUnjudgedTrial);
elements.evalExperiment.addEventListener("change", () => {
  activeEvaluationId = elements.evalExperiment.value || null;
  currentEvaluationTrialId = null;
  saveEvaluationStore();
  renderEvaluationSummary();
  showNextUnjudgedTrial();
});
elements.evalExportJson.addEventListener("click", exportEvaluationJson);
elements.evalExportCsv.addEventListener("click", exportEvaluationCsv);
elements.evalDelete.addEventListener("click", deleteActiveEvaluation);
window.addEventListener("resize", drawPlot);

renderEvaluationSelector();
renderEvaluationSummary();
showNextUnjudgedTrial();
updateControlLabels();
drawPlot();
poll();
