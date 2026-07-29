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
  "evalSummary", "evalExperiment", "evalSummaryBody", "evalExportJson", "evalExportCsv", "evalDelete",
  "distTask", "distTaskKind", "distTaskDescription", "distTargetBody", "distProfileA", "distDiversityA", "distCapA",
  "distFloorA", "distGuardA", "distProfileB", "distDiversityB", "distCapB", "distFloorB", "distGuardB", "distBlocks",
  "distDraws", "distSeed", "distLockSupport", "distAutoPrefix", "distPrefix", "distStart", "distStop", "distClear",
  "distRunStatus", "distProgress", "distWarnings", "distResults", "distResultCaption", "distExportJson", "distExportCsv",
  "distSummaryBody",
  "distChartTarget", "distChartRaw", "distChartA", "distChartB", "distChartTitleA", "distChartTitleB", "distMassBars",
  "distOutcomeBody", "distMappingBody", "distInvalidTokens"
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
let distributionRunning = false;
let distributionCancelRequested = false;
const maximumDisplayPointCount = 64;
const evaluationStorageKey = "logit-scope-evaluations-v1";
const evaluationActiveKey = "logit-scope-active-evaluation-v1";
const distributionStorageKey = "logit-scope-distributions-v1";
let evaluationStore = loadEvaluationStore();
let activeEvaluationId = loadActiveEvaluationId();
let activeDistributionRun = loadDistributionRun();

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
  if (diversity !== 1) return `${diversity.toFixed(2)}× alternatives`;
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
  updateTranscript(evaluationRunning ? "Blind comparison in progress. Responses stay concealed until each pair is ready." :
    distributionRunning ? "Distribution Lab is measuring exact next-token probabilities in an isolated context." : next.transcript || "");

  const canSend = next.modelLoaded && !next.generating && !evaluationRunning && !distributionRunning;
  elements.send.disabled = !canSend;
  elements.message.disabled = !canSend;
  elements.stop.disabled = !next.generating || evaluationRunning || distributionRunning;
  elements.clear.disabled = !next.modelLoaded || evaluationRunning || distributionRunning;
  elements.evalStart.disabled = evaluationRunning || distributionRunning || !next.modelLoaded || next.generating;
  elements.distStart.disabled = distributionRunning || evaluationRunning || !next.modelLoaded || next.generating;
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
    temperature: "Temperature",
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
  elements.distStart.disabled = running || distributionRunning || !snapshot?.modelLoaded || snapshot?.generating;
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
  if (evaluationRunning || distributionRunning) return;
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

const distributionTasks = [
  {
    id: "two-dice-sum",
    name: "Sum of two fair six-sided dice",
    kind: "implicit",
    description: "Two independent fair six-sided dice are rolled. The outcome is their sum.",
    outcomes: [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1].map((weight, index) =>
      ({ id: `sum-${index + 2}`, text: `sum ${index + 2}`, weight }))
  },
  {
    id: "four-coin-heads",
    name: "Number of heads in four fair tosses",
    kind: "implicit",
    description: "A fair coin is tossed independently four times. The outcome is the total number of heads.",
    outcomes: [1, 4, 6, 4, 1].map((weight, heads) =>
      ({ id: `heads-${heads}`, text: `${heads} ${heads === 1 ? "head" : "heads"}`, weight }))
  },
  {
    id: "two-dice-maximum",
    name: "Maximum of two fair six-sided dice",
    kind: "implicit",
    description: "Two independent fair six-sided dice are rolled. The outcome is the larger of the two values.",
    outcomes: [1, 3, 5, 7, 9, 11].map((weight, index) =>
      ({ id: `maximum-${index + 1}`, text: `maximum ${index + 1}`, weight }))
  },
  {
    id: "first-head-three-tosses",
    name: "First head position in three fair tosses",
    kind: "implicit",
    description: "A fair coin is tossed up to three times. The outcome is the toss on which the first head appears, or no heads.",
    outcomes: [
      { id: "head-1", text: "head on toss 1", weight: 4 },
      { id: "head-2", text: "head on toss 2", weight: 2 },
      { id: "head-3", text: "head on toss 3", weight: 1 },
      { id: "no-heads", text: "no heads", weight: 1 }
    ]
  },
  {
    id: "explicit-five-way-control",
    name: "Explicit 8:4:2:1:1 control",
    kind: "control",
    description: "A calibrated source emits outcome one, two, three, four, or five with relative frequencies 8, 4, 2, 1, and 1 respectively.",
    outcomes: [8, 4, 2, 1, 1].map((weight, index) =>
      ({ id: `control-${index + 1}`, text: `outcome ${index + 1}`, weight }))
  }
];

function loadDistributionRun() {
  try {
    const run = JSON.parse(window.localStorage.getItem(distributionStorageKey));
    if (run?.version === 1 && Array.isArray(run.mappings) && Array.isArray(run.probes)) {
      if (run.status === "running") run.status = "paused";
      if (run.autoSelectAssistantPrefix === undefined) run.autoSelectAssistantPrefix = true;
      run.nextMappingIndex = Math.max(run.nextMappingIndex || 0, run.probes.length);
      return run;
    }
  } catch {
    // A malformed or unavailable local store should not block the application.
  }
  return null;
}

function saveDistributionRun() {
  try {
    if (activeDistributionRun)
      window.localStorage.setItem(distributionStorageKey, JSON.stringify(activeDistributionRun));
    else
      window.localStorage.removeItem(distributionStorageKey);
  } catch (error) {
    elements.error.textContent = `Distribution results could not be saved: ${error.message}`;
  }
}

function selectedDistributionTask() {
  return distributionTasks.find(task => task.id === elements.distTask.value) || distributionTasks[0];
}

function normalizedTaskOutcomes(task) {
  const total = task.outcomes.reduce((sum, outcome) => sum + outcome.weight, 0);
  return task.outcomes.map(outcome => ({ ...outcome, targetProbability: outcome.weight / total }));
}

function renderDistributionTask() {
  const task = selectedDistributionTask();
  elements.distTaskKind.textContent = task.kind === "control" ? "Explicit control" : "Implicit probability task";
  elements.distTaskKind.className = `distribution-kind${task.kind === "control" ? " control" : ""}`;
  elements.distTaskDescription.textContent = task.description;
  elements.distTargetBody.replaceChildren();
  for (const outcome of normalizedTaskOutcomes(task)) {
    const row = document.createElement("tr");
    appendEvaluationCell(row, outcome.text);
    appendEvaluationCell(row, String(outcome.weight));
    appendEvaluationCell(row, percentage(outcome.targetProbability, 2));
    elements.distTargetBody.append(row);
  }
}

function createSeededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function distributionGreatestCommonDivisor(a, b) {
  let lhs = Math.abs(a);
  let rhs = Math.abs(b);
  while (rhs !== 0) {
    const next = lhs % rhs;
    lhs = rhs;
    rhs = next;
  }
  return lhs;
}

function distributionShuffle(values, random) {
  const result = values.slice();
  for (let index = result.length - 1; index > 0; --index) {
    const other = Math.floor(random() * (index + 1));
    [result[index], result[other]] = [result[other], result[index]];
  }
  return result;
}

function distributionCoprimeStride(size, random) {
  if (size <= 2) return 1;
  const candidates = [];
  for (let stride = 1; stride < size; ++stride) {
    if (distributionGreatestCommonDivisor(stride, size) === 1) candidates.push(stride);
  }
  return candidates[Math.floor(random() * candidates.length)];
}

function createBalancedDistributionMappings(outcomes, blockCount, seed) {
  const labels = Array.from({ length: outcomes.length }, (_, index) => String.fromCharCode(65 + index));
  const mappings = [];
  const size = outcomes.length;
  for (let block = 0; block < blockCount; ++block) {
    const random = createSeededRandom((seed + block) >>> 0);
    const baseOutcomes = distributionShuffle(outcomes, random);
    const baseLabels = distributionShuffle(labels, random);
    const stride = distributionCoprimeStride(size, random);
    for (let rotation = 0; rotation < size; ++rotation) {
      const assignedLabelByOutcomeId = new Map();
      for (let index = 0; index < size; ++index)
        assignedLabelByOutcomeId.set(outcomes[index].id, baseLabels[(index + rotation) % size]);
      const displayOffset = (rotation * stride) % size;
      const rows = [];
      for (let position = 0; position < size; ++position) {
        const outcome = baseOutcomes[(position + displayOffset) % size];
        rows.push({ position, outcomeId: outcome.id, label: assignedLabelByOutcomeId.get(outcome.id) });
      }
      mappings.push({ id: `${block}:${rotation}`, block, rotation, stride, rows });
    }
  }
  return mappings;
}

function verifyBalancedDistributionBlock(mappings, outcomes, block) {
  const blockMappings = mappings.filter(mapping => mapping.block === block);
  if (blockMappings.length !== outcomes.length) throw new Error("Balanced block has the wrong number of mappings");
  const expectedLabels = outcomes.length;
  for (const outcome of outcomes) {
    const labels = new Set();
    const positions = new Set();
    for (const mapping of blockMappings) {
      const row = mapping.rows.find(value => value.outcomeId === outcome.id);
      if (!row) throw new Error(`Mapping omitted outcome ${outcome.id}`);
      labels.add(row.label);
      positions.add(row.position);
    }
    if (labels.size !== expectedLabels) throw new Error(`Outcome ${outcome.id} did not receive every label`);
    if (positions.size !== outcomes.length) throw new Error(`Outcome ${outcome.id} did not occupy every position`);
  }
}

function distributionField(side, name) {
  return elements[`dist${name}${side}`];
}

function distributionSettings(side, seed = Number(elements.distSeed.value) || 0) {
  return {
    profile: distributionField(side, "Profile").value,
    diversity: Math.max(0, Math.min(2, Number(distributionField(side, "Diversity").value) / 100)),
    candidateCap: Number(distributionField(side, "Cap").value),
    minimumRelativeProbability: Number(distributionField(side, "Floor").value),
    seed,
    protectControlTokens: distributionField(side, "Guard").checked
  };
}

function updateDistributionConfigControls() {
  for (const side of ["A", "B"])
    distributionField(side, "Diversity").disabled = distributionRunning || distributionField(side, "Profile").value === "none";
  const supportLocked = elements.distLockSupport.checked;
  for (const name of ["Cap", "Floor", "Guard"]) distributionField("B", name).disabled = distributionRunning || supportLocked;
}

function synchronizeDistributionSupport() {
  if (!elements.distLockSupport.checked) {
    updateDistributionConfigControls();
    return;
  }
  distributionField("B", "Cap").value = distributionField("A", "Cap").value;
  distributionField("B", "Floor").value = distributionField("A", "Floor").value;
  distributionField("B", "Guard").checked = distributionField("A", "Guard").checked;
  updateDistributionConfigControls();
}

function buildDistributionPrompt(task, mapping) {
  const outcomes = new Map(task.outcomes.map(outcome => [outcome.id, outcome]));
  const mappingLines = mapping.rows.map(row => `${row.label} = ${outcomes.get(row.outcomeId).text}`).join("\n");
  return `${task.description}

The possible outcomes are represented by opaque labels:
${mappingLines}

Simulate one independent occurrence of this process and return the label for its outcome. Sample according to the true probability distribution implied by the process. Do not merely choose the most likely outcome. Do not explain, show calculations, list probabilities, or add punctuation.

Return exactly one label.`;
}

function createDistributionRun() {
  const task = selectedDistributionTask();
  const outcomes = normalizedTaskOutcomes(task);
  const blocks = Math.max(1, Math.min(10, Number(elements.distBlocks.value) || 1));
  const sampleCount = Math.max(0, Math.min(1000000, Number(elements.distDraws.value) || 0));
  const seed = Math.max(0, Math.min(4294967295, Number(elements.distSeed.value) || 0));
  const assistantPrefix = elements.distPrefix.value;
  const mappings = createBalancedDistributionMappings(outcomes, blocks, seed);
  for (let block = 0; block < blocks; ++block) verifyBalancedDistributionBlock(mappings, outcomes, block);
  const settingsA = distributionSettings("A", seed);
  const settingsB = distributionSettings("B", seed);
  return {
    version: 1,
    id: createEvaluationId(),
    startedAt: new Date().toISOString(),
    completedAt: null,
    status: "running",
    error: null,
    task: { ...task, outcomes },
    blocks,
    sampleCount,
    seed,
    assistantPrefix,
    autoSelectAssistantPrefix: elements.distAutoPrefix.checked,
    supportLocked: elements.distLockSupport.checked,
    configurations: [
      { id: "A", name: profileDisplayName(settingsA.profile), settings: settingsA },
      { id: "B", name: profileDisplayName(settingsB.profile), settings: settingsB }
    ],
    mappings,
    nextMappingIndex: 0,
    probes: []
  };
}

function setDistributionRunning(running) {
  distributionRunning = running;
  const setupControls = [
    "distTask", "distProfileA", "distDiversityA", "distCapA", "distFloorA", "distGuardA", "distProfileB",
    "distDiversityB", "distCapB", "distFloorB", "distGuardB", "distBlocks", "distDraws", "distSeed",
    "distLockSupport", "distAutoPrefix", "distPrefix"
  ];
  const primaryControls = [
    "profile", "diversity", "candidateCap", "minimumRelativeProbability", "seed", "protectControlTokens", "message",
    "clear", "send", "evalStart"
  ];
  for (const id of [...setupControls, ...primaryControls]) elements[id].disabled = running;
  elements.distStart.disabled = running || evaluationRunning || !snapshot?.modelLoaded || snapshot?.generating;
  elements.distStop.disabled = !running;
  elements.distClear.disabled = running || !activeDistributionRun;
  if (!running) {
    updateControlLabels();
    updateDistributionConfigControls();
    if (snapshot) applySnapshot(snapshot);
  }
}

async function submitDistributionMapping(run, mapping) {
  const outcomesById = new Map(run.task.outcomes.map(outcome => [outcome.id, outcome]));
  const labelsByOutcomeId = new Map(mapping.rows.map(row => [row.outcomeId, row.label]));
  const mappingId = `${run.task.id}:block-${mapping.block}:rotation-${mapping.rotation}`;
  const accepted = await request("/api/distribution", {
    method: "POST",
    body: JSON.stringify({
      prompt: buildDistributionPrompt(run.task, mapping),
      assistantPrefix: run.assistantPrefix,
      autoSelectAssistantPrefix: run.autoSelectAssistantPrefix,
      mappingId,
      sampleCount: run.sampleCount,
      seed: run.seed,
      outcomes: run.task.outcomes.map(outcome => ({
        id: outcome.id,
        text: outcome.text,
        label: labelsByOutcomeId.get(outcome.id),
        targetProbability: outcomesById.get(outcome.id).targetProbability
      })),
      configurations: run.configurations
    })
  });
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    if (distributionCancelRequested) throw new Error("Distribution run stopped");
    const result = await request("/api/distribution");
    if (result.id === accepted.id && result.ready) return result;
    await evaluationDelay(85);
  }
  throw new Error("Distribution probe timed out");
}

async function startDistributionRun() {
  if (distributionRunning || evaluationRunning) return;
  if (!snapshot?.modelLoaded || snapshot.generating) {
    elements.error.textContent = "The model must be ready before starting Distribution Lab.";
    return;
  }
  if (!elements.distPrefix.value) {
    elements.error.textContent = "Assistant prefix must not be empty.";
    return;
  }

  const resumable = activeDistributionRun && ["paused", "stopped"].includes(activeDistributionRun.status) &&
    activeDistributionRun.nextMappingIndex < activeDistributionRun.mappings.length;
  if (!resumable) activeDistributionRun = createDistributionRun();
  activeDistributionRun.status = "running";
  activeDistributionRun.error = null;
  distributionCancelRequested = false;
  saveDistributionRun();
  setDistributionRunning(true);
  renderDistributionResults();
  elements.error.textContent = "";

  try {
    while (activeDistributionRun.nextMappingIndex < activeDistributionRun.mappings.length) {
      if (distributionCancelRequested) throw new Error("Distribution run stopped");
      const mappingIndex = activeDistributionRun.nextMappingIndex;
      const mapping = activeDistributionRun.mappings[mappingIndex];
      elements.distRunStatus.textContent =
        `Measuring block ${mapping.block + 1}, rotation ${mapping.rotation + 1}…`;
      elements.distProgress.textContent = `${mappingIndex + 1} / ${activeDistributionRun.mappings.length} probes`;
      const result = await submitDistributionMapping(activeDistributionRun, mapping);
      if (result.status !== "Complete") {
        activeDistributionRun.failedProbe = { mapping, result };
        throw new Error(result.status);
      }
      activeDistributionRun.probes.push({ mapping, result });
      activeDistributionRun.nextMappingIndex = mappingIndex + 1;
      saveDistributionRun();
      renderDistributionResults();
    }
    activeDistributionRun.status = "complete";
    activeDistributionRun.completedAt = new Date().toISOString();
    elements.distRunStatus.textContent = "Balanced distribution run complete.";
    elements.distProgress.textContent = `${activeDistributionRun.probes.length} exact shared-logits probes`;
  } catch (error) {
    activeDistributionRun.status = distributionCancelRequested ? "stopped" : "error";
    activeDistributionRun.error = error.message;
    elements.distRunStatus.textContent = distributionCancelRequested ? "Run stopped; completed probes were saved." : "Distribution run failed.";
    elements.distProgress.textContent = `${activeDistributionRun.probes.length} / ${activeDistributionRun.mappings.length} probes complete`;
    if (!distributionCancelRequested) elements.error.textContent = error.message;
  } finally {
    saveDistributionRun();
    setDistributionRunning(false);
    renderDistributionResults();
  }
}

function stopDistributionRun() {
  if (!distributionRunning) return;
  distributionCancelRequested = true;
  elements.distRunStatus.textContent = "Stopping the active probe…";
  request("/api/stop", { method: "POST", body: "{}" }).catch(error => elements.error.textContent = error.message);
}

function clearDistributionRun() {
  if (distributionRunning || !activeDistributionRun) return;
  if (!window.confirm("Clear the saved Distribution Lab run from this browser?")) return;
  activeDistributionRun = null;
  saveDistributionRun();
  renderDistributionResults();
}

function distributionNormalize(values) {
  const total = values.reduce((sum, value) => sum + value, 0);
  return total > 0 ? values.map(value => value / total) : null;
}

function distributionEntropy(values) {
  const normalized = distributionNormalize(values);
  if (!normalized) return null;
  return -normalized.reduce((sum, value) => sum + (value > 0 ? value * Math.log(value) : 0), 0);
}

function distributionTotalVariation(left, right) {
  const lhs = distributionNormalize(left);
  const rhs = distributionNormalize(right);
  if (!lhs || !rhs || lhs.length !== rhs.length) return null;
  return 0.5 * lhs.reduce((sum, value, index) => sum + Math.abs(value - rhs[index]), 0);
}

function distributionJensenShannon(left, right) {
  const lhs = distributionNormalize(left);
  const rhs = distributionNormalize(right);
  if (!lhs || !rhs || lhs.length !== rhs.length) return null;
  let divergence = 0;
  for (let index = 0; index < lhs.length; ++index) {
    const midpoint = 0.5 * (lhs[index] + rhs[index]);
    if (lhs[index] > 0) divergence += 0.5 * lhs[index] * Math.log(lhs[index] / midpoint);
    if (rhs[index] > 0) divergence += 0.5 * rhs[index] * Math.log(rhs[index] / midpoint);
  }
  return divergence;
}

function distributionPairwiseOrderAccuracy(predicted, target) {
  const normalizedPredicted = distributionNormalize(predicted);
  const normalizedTarget = distributionNormalize(target);
  if (!normalizedPredicted || !normalizedTarget) return null;
  let score = 0;
  let pairs = 0;
  for (let left = 0; left < target.length; ++left) {
    for (let right = left + 1; right < target.length; ++right) {
      const targetDifference = normalizedTarget[left] - normalizedTarget[right];
      if (targetDifference === 0) continue;
      const predictedDifference = normalizedPredicted[left] - normalizedPredicted[right];
      ++pairs;
      if (predictedDifference === 0) score += 0.5;
      else if ((predictedDifference > 0) === (targetDifference > 0)) score += 1;
    }
  }
  return pairs ? score / pairs : 1;
}

function distributionStandardDeviation(values) {
  if (!values.length) return null;
  const mean = average(values);
  return Math.sqrt(average(values.map(value => Math.pow(value - mean, 2))));
}

function distributionStageDescriptors(run) {
  const configA = run.configurations.find(configuration => configuration.id === "A");
  const configB = run.configurations.find(configuration => configuration.id === "B");
  const configuration = (probe, id) => probe.result.configurations.find(value => value.id === id);
  return [
    { key: "fullRaw", label: "Full raw model", className: "full-raw", stage: probe => probe.result.fullRaw },
    {
      key: "retainedA", label: `Retained raw A · ${configA.name}`, className: "config-a",
      configurationId: "A", stage: probe => configuration(probe, "A")?.retainedRaw
    },
    {
      key: "shapedA", label: `Shaped A · ${configA.name}`, className: "config-a",
      configurationId: "A", shaped: true, stage: probe => configuration(probe, "A")?.shaped
    },
    {
      key: "retainedB", label: `Retained raw B · ${configB.name}`, className: "config-b",
      configurationId: "B", stage: probe => configuration(probe, "B")?.retainedRaw
    },
    {
      key: "shapedB", label: `Shaped B · ${configB.name}`, className: "config-b",
      configurationId: "B", shaped: true, stage: probe => configuration(probe, "B")?.shaped
    }
  ];
}

function aggregateDistributionStage(run, descriptor) {
  const stages = run.probes.map(probe => descriptor.stage(probe)).filter(Boolean);
  if (!stages.length) return null;
  const target = run.task.outcomes.map(outcome => outcome.targetProbability);
  const pooled = run.task.outcomes.map(outcome =>
    average(stages.map(stage => stage.outcomes.find(value => value.id === outcome.id)?.probability || 0)));
  const validMass = pooled.reduce((sum, value) => sum + value, 0);
  const invalidMass = Math.max(0, 1 - validMass);
  const conditional = distributionNormalize(pooled);
  const targetEntropy = distributionEntropy(target);
  const openPredicted = [...pooled, invalidMass];
  const openTarget = [...target, 0];
  const perMappingOpenTv = stages.map(stage => stage.openMetrics.totalVariation);
  const perMappingConditionalTv = stages.filter(stage => stage.conditionalMetrics.valid)
    .map(stage => stage.conditionalMetrics.totalVariation);
  const conditionalVectors = stages.map(stage => distributionNormalize(
    run.task.outcomes.map(outcome => stage.outcomes.find(value => value.id === outcome.id)?.probability || 0))).filter(Boolean);
  const mappingDivergences = conditional && conditionalVectors.length ?
    conditionalVectors.map(values => distributionJensenShannon(values, conditional)).filter(Number.isFinite) : [];
  const mappingSensitivityDivergence = average(mappingDivergences);
  return {
    descriptor,
    stages,
    pooled,
    conditional,
    projected: run.task.outcomes.map(outcome => stages.reduce((sum, stage) =>
      sum + (stage.outcomes.find(value => value.id === outcome.id)?.projectedCount || 0), 0)),
    validMass,
    invalidMass,
    openTv: distributionTotalVariation(openPredicted, openTarget),
    conditionalTv: conditional ? distributionTotalVariation(conditional, target) : null,
    openJsd: distributionJensenShannon(openPredicted, openTarget),
    openJsDistance: Math.sqrt((distributionJensenShannon(openPredicted, openTarget) || 0) / Math.log(2)),
    orderAccuracy: conditional ? distributionPairwiseOrderAccuracy(conditional, target) : null,
    entropyError: conditional ? distributionEntropy(conditional) - targetEntropy : null,
    missingTargetMass: average(stages.map(stage => stage.openMetrics.missingTargetMass)),
    meanMappingOpenTv: average(perMappingOpenTv),
    standardDeviationOpenTv: distributionStandardDeviation(perMappingOpenTv),
    meanMappingConditionalTv: average(perMappingConditionalTv),
    standardDeviationConditionalTv: distributionStandardDeviation(perMappingConditionalTv),
    mappingSensitivityDivergence,
    mappingSensitivity: mappingSensitivityDivergence === null ? null :
      Math.sqrt(mappingSensitivityDivergence / Math.log(2))
  };
}

function distributionAggregates(run) {
  return distributionStageDescriptors(run).map(descriptor => aggregateDistributionStage(run, descriptor)).filter(Boolean);
}

function formatDistributionMetric(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function appendDistributionSummaryRow(aggregate) {
  const row = document.createElement("tr");
  row.className = aggregate.descriptor.className;
  appendEvaluationCell(row, aggregate.descriptor.label);
  appendEvaluationCell(row, formatDistributionMetric(aggregate.openTv));
  appendEvaluationCell(row, formatDistributionMetric(aggregate.conditionalTv));
  appendEvaluationCell(row, formatDistributionMetric(aggregate.openJsDistance));
  appendEvaluationCell(row, `${percentage(aggregate.validMass)} / ${percentage(aggregate.invalidMass)}`);
  appendEvaluationCell(row, percentage(aggregate.missingTargetMass));
  appendEvaluationCell(row, percentage(aggregate.orderAccuracy));
  appendEvaluationCell(row, Number.isFinite(aggregate.entropyError) ? `${aggregate.entropyError >= 0 ? "+" : ""}${aggregate.entropyError.toFixed(4)}` : "—");
  appendEvaluationCell(row,
    `${formatDistributionMetric(aggregate.meanMappingOpenTv)} ± ${formatDistributionMetric(aggregate.standardDeviationOpenTv)} / ` +
    `${formatDistributionMetric(aggregate.meanMappingConditionalTv)} ± ${formatDistributionMetric(aggregate.standardDeviationConditionalTv)}`);
  appendEvaluationCell(row, formatDistributionMetric(aggregate.mappingSensitivity));
  elements.distSummaryBody.append(row);
}

function drawDistributionSeriesChart(canvas, run, values, color, maximum, emptyMessage) {
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
  context.clearRect(0, 0, w, h);
  if (!run) return;

  const margin = { left: 42, right: 8, top: 16, bottom: 66 };
  const plotWidth = Math.max(1, w - margin.left - margin.right);
  const plotHeight = Math.max(1, h - margin.top - margin.bottom);
  context.font = "10px ui-sans-serif, system-ui";
  for (let step = 0; step <= 4; ++step) {
    const probability = maximum * step / 4;
    const y = margin.top + plotHeight - probability / maximum * plotHeight;
    context.strokeStyle = step === 0 ? "#45515e" : "#27313a";
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(w - margin.right, y);
    context.stroke();
    context.fillStyle = "#7f8d9a";
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(`${(probability * 100).toFixed(0)}%`, margin.left - 7, y);
  }
  const groupWidth = plotWidth / run.task.outcomes.length;
  const barWidth = Math.max(3, Math.min(28, groupWidth * 0.56));
  const hasValues = values.length === run.task.outcomes.length;
  for (let outcomeIndex = 0; outcomeIndex < run.task.outcomes.length; ++outcomeIndex) {
    const groupCenter = margin.left + groupWidth * (outcomeIndex + 0.5);
    if (hasValues) {
      const value = values[outcomeIndex] || 0;
      const barHeight = value / maximum * plotHeight;
      context.fillStyle = color;
      context.fillRect(groupCenter - barWidth / 2, margin.top + plotHeight - barHeight, barWidth, barHeight);
      if (groupWidth >= 42 && value > 0) {
        context.fillStyle = "#c5d0da";
        context.textAlign = "center";
        context.textBaseline = "bottom";
        context.font = "9px ui-sans-serif, system-ui";
        context.fillText(`${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`, groupCenter,
                         Math.max(margin.top + 9, margin.top + plotHeight - barHeight - 3));
        context.font = "10px ui-sans-serif, system-ui";
      }
    }
    context.save();
    context.translate(groupCenter + 2, h - margin.bottom + 8);
    context.rotate(-Math.PI / 4);
    context.fillStyle = "#91a0ae";
    context.textAlign = "right";
    context.textBaseline = "middle";
    const label = run.task.outcomes[outcomeIndex].text;
    context.fillText(label.length > 18 ? `${label.slice(0, 17)}…` : label, 0, 0);
    context.restore();
  }
  if (!hasValues && emptyMessage) {
    context.fillStyle = "#c9985a";
    context.font = "11px ui-sans-serif, system-ui";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(emptyMessage, margin.left + plotWidth / 2, margin.top + plotHeight / 2, plotWidth - 24);
  }
}

function drawDistributionCharts(run, aggregates) {
  const byKey = new Map(aggregates.map(aggregate => [aggregate.descriptor.key, aggregate]));
  const target = run ? run.task.outcomes.map(outcome => outcome.targetProbability) : [];
  const configurationA = run?.configurations.find(configuration => configuration.id === "A");
  const configurationB = run?.configurations.find(configuration => configuration.id === "B");
  elements.distChartTitleA.textContent = configurationA ? `A shaped conditional · ${configurationA.name}` : "A shaped conditional";
  elements.distChartTitleB.textContent = configurationB ? `B shaped conditional · ${configurationB.name}` : "B shaped conditional";
  const charts = [
    { canvas: elements.distChartTarget, values: target, color: "#b779ff" },
    { canvas: elements.distChartRaw, values: byKey.get("fullRaw")?.conditional || [], color: "#43d0c1" },
    { canvas: elements.distChartA, values: byKey.get("shapedA")?.conditional || [], color: "#ffd07a" },
    { canvas: elements.distChartB, values: byKey.get("shapedB")?.conditional || [], color: "#ff9b42" }
  ];
  const rawMaximum = Math.max(0.05, ...charts.flatMap(chart => chart.values));
  const maximum = Math.ceil(rawMaximum * 20) / 20;
  for (const chart of charts)
    drawDistributionSeriesChart(chart.canvas, run, chart.values, chart.color, maximum,
                                "No valid label mass after support filtering");
}

function renderDistributionMassBars(aggregates) {
  elements.distMassBars.replaceChildren();
  for (const key of ["fullRaw", "shapedA", "shapedB"]) {
    const aggregate = aggregates.find(value => value.descriptor.key === key);
    if (!aggregate) continue;
    const row = document.createElement("div");
    row.className = "distribution-mass-row";
    const label = document.createElement("span");
    label.textContent = aggregate.descriptor.label;
    const track = document.createElement("div");
    track.className = "distribution-mass-track";
    const valid = document.createElement("span");
    valid.className = "distribution-mass-valid";
    valid.style.width = `${Math.max(0, Math.min(100, aggregate.validMass * 100))}%`;
    track.append(valid);
    const value = document.createElement("strong");
    value.textContent = `${percentage(aggregate.validMass)} valid`;
    row.append(label, track, value);
    elements.distMassBars.append(row);
  }
}

function renderDistributionOutcomeTable(run, aggregates) {
  elements.distOutcomeBody.replaceChildren();
  const byKey = new Map(aggregates.map(aggregate => [aggregate.descriptor.key, aggregate]));
  for (let index = 0; index < run.task.outcomes.length; ++index) {
    const outcome = run.task.outcomes[index];
    const row = document.createElement("tr");
    appendEvaluationCell(row, outcome.text);
    appendEvaluationCell(row, percentage(outcome.targetProbability, 2));
    appendEvaluationCell(row, percentage(byKey.get("fullRaw")?.pooled[index]));
    appendEvaluationCell(row, percentage(byKey.get("shapedA")?.pooled[index]));
    appendEvaluationCell(row, percentage(byKey.get("shapedB")?.pooled[index]));
    appendEvaluationCell(row, (byKey.get("shapedA")?.projected[index] || 0).toLocaleString());
    appendEvaluationCell(row, (byKey.get("shapedB")?.projected[index] || 0).toLocaleString());
    elements.distOutcomeBody.append(row);
  }
}

function renderDistributionMappingTable(run) {
  elements.distMappingBody.replaceChildren();
  const invalidLines = [];
  for (const probe of run.probes) {
    const configA = probe.result.configurations.find(configuration => configuration.id === "A");
    const configB = probe.result.configurations.find(configuration => configuration.id === "B");
    for (const fullOutcome of probe.result.fullRaw.outcomes) {
      const outcomeA = configA?.shaped.outcomes.find(value => value.id === fullOutcome.id);
      const outcomeB = configB?.shaped.outcomes.find(value => value.id === fullOutcome.id);
      const row = document.createElement("tr");
      appendEvaluationCell(row, probe.result.mappingId);
      appendEvaluationCell(row, fullOutcome.text);
      appendEvaluationCell(row, fullOutcome.label);
      appendEvaluationCell(row, String(fullOutcome.tokenId));
      appendEvaluationCell(row, percentage(fullOutcome.probability));
      appendEvaluationCell(row, percentage(outcomeA?.probability));
      appendEvaluationCell(row, percentage(outcomeB?.probability));
      appendEvaluationCell(row, `${outcomeA?.retained ? "yes" : "no"} / ${outcomeB?.retained ? "yes" : "no"}`);
      elements.distMappingBody.append(row);
    }
    const tokenText = stage => stage.topInvalidTokens.slice(0, 5)
      .map(token => `${JSON.stringify(token.text)} (${percentage(token.probability)}, rank ${token.rank + 1})`).join(", ");
    invalidLines.push(`${probe.result.mappingId}
  full raw: ${tokenText(probe.result.fullRaw)}
  A shaped: ${tokenText(configA.shaped)}
  B shaped: ${tokenText(configB.shaped)}`);
  }
  elements.distInvalidTokens.textContent = invalidLines.length ?
    `Highest-probability invalid tokens\n\n${invalidLines.join("\n\n")}` : "";
}

function distributionSupportMatches(run) {
  const [left, right] = run.configurations.map(configuration => configuration.settings);
  return left.candidateCap === right.candidateCap &&
    Math.abs(left.minimumRelativeProbability - right.minimumRelativeProbability) < 1e-12 &&
    left.protectControlTokens === right.protectControlTokens;
}

function distributionWarnings(run, aggregates) {
  const warnings = [];
  if (run.error) warnings.push(run.error);
  if (run.probes.length < run.mappings.length)
    warnings.push(`Partial aggregate: ${run.probes.length} of ${run.mappings.length} balanced mappings are complete.`);
  if (!distributionSupportMatches(run))
    warnings.push("A and B use different candidate cap, Min-P, or protocol-guard settings; the comparison mixes support and shaping effects.");
  const automaticBoundaries = [...new Map(run.probes
    .filter(probe => probe.result.assistantPrefixAutoSelected)
    .map(probe => {
      const assistantPrefix = probe.result.assistantPrefix;
      const labelTokenPrefix = probe.result.labelTokenPrefix || "";
      return [`${assistantPrefix}\u0000${labelTokenPrefix}`, { assistantPrefix, labelTokenPrefix }];
    })).values()];
  if (automaticBoundaries.length) {
    const descriptions = automaticBoundaries.map(boundary =>
      `context ${JSON.stringify(boundary.assistantPrefix)} + label-token prefix ${JSON.stringify(boundary.labelTokenPrefix)}`);
    warnings.push(`Tokenizer-compatible boundary selected automatically: ${descriptions.join(", ")}.`);
  }
  if (aggregates.some(aggregate => aggregate.missingTargetMass > 1e-9))
    warnings.push("At least one valid outcome was pruned. Missing target mass contributes directly to the reported diagnostic.");
  if (aggregates.some(aggregate => aggregate.validMass < 0.5))
    warnings.push("Valid label mass is below 50% in at least one stage; conditional weighting may look good despite protocol failure.");
  const configurations = run.probes.flatMap(probe => probe.result.configurations);
  if (configurations.some(configuration => configuration.diagnostics.targetSaturated))
    warnings.push("At least one sampler entropy target saturated at the retained support limit.");
  if (configurations.some(configuration => Math.abs(configuration.diagnostics.samplerEntropyError) > 0.02))
    warnings.push("Protocol-guard projection materially changed final sampler entropy in at least one probe.");
  const sensitive = aggregates.filter(aggregate => aggregate.mappingSensitivity > 0.1);
  if (sensitive.length)
    warnings.push("Mapping sensitivity is high in at least one stage; pooled label rotations may be hiding unstable individual prompts.");
  if (aggregates.some(aggregate => aggregate.openTv < 0.1 && aggregate.mappingSensitivity > 0.1))
    warnings.push("A pooled score looks strong while per-mapping distributions remain unstable.");

  if (distributionSupportMatches(run)) {
    const retainedMismatch = run.probes.some(probe => {
      const left = probe.result.configurations.find(configuration => configuration.id === "A")?.retainedRaw;
      const right = probe.result.configurations.find(configuration => configuration.id === "B")?.retainedRaw;
      return left && right && left.outcomes.some((outcome, index) =>
        outcome.retained !== right.outcomes[index].retained || Math.abs(outcome.probability - right.outcomes[index].probability) > 1e-9);
    });
    if (retainedMismatch) warnings.push("Retained-raw A and B differ even though support settings match; inspect the exported probe data.");
  }
  return warnings;
}

function renderDistributionResults() {
  const run = activeDistributionRun;
  elements.distClear.disabled = distributionRunning || !run;
  elements.distExportJson.disabled = !run?.probes.length;
  elements.distExportCsv.disabled = !run?.probes.length;
  elements.distSummaryBody.replaceChildren();
  if (!run) {
    elements.distResults.hidden = true;
    elements.distWarnings.hidden = true;
    elements.distWarnings.replaceChildren();
    elements.distRunStatus.textContent = "Ready to measure.";
    elements.distProgress.textContent = "";
    elements.distStart.textContent = "Run balanced block";
    drawDistributionCharts(null, []);
    return;
  }

  elements.distStart.textContent = ["paused", "stopped"].includes(run.status) &&
    run.nextMappingIndex < run.mappings.length ? "Resume saved run" : "Run another balanced block";
  if (!distributionRunning) {
    if (run.status === "complete") {
      elements.distRunStatus.textContent = "Saved balanced run complete.";
      elements.distProgress.textContent = `${run.probes.length} probes`;
    } else if (run.status === "paused") {
      elements.distRunStatus.textContent = "Saved run paused after browser reload.";
      elements.distProgress.textContent = `${run.probes.length} / ${run.mappings.length} probes complete`;
    } else if (run.status === "stopped") {
      elements.distRunStatus.textContent = "Saved run stopped and resumable.";
      elements.distProgress.textContent = `${run.probes.length} / ${run.mappings.length} probes complete`;
    } else if (run.status === "error") {
      elements.distRunStatus.textContent = "Saved run ended with an error.";
      elements.distProgress.textContent = `${run.probes.length} / ${run.mappings.length} probes complete`;
    }
  }

  const aggregates = distributionAggregates(run);
  elements.distResults.hidden = !aggregates.length;
  if (aggregates.length) {
    for (const aggregate of aggregates) appendDistributionSummaryRow(aggregate);
    const partial = run.probes.length < run.mappings.length ? "Partial" : "Complete";
    elements.distResultCaption.textContent =
      `${partial} semantic aggregate across ${run.probes.length}/${run.mappings.length} mappings · ${run.task.name}.`;
    renderDistributionOutcomeTable(run, aggregates);
    renderDistributionMappingTable(run);
    renderDistributionMassBars(aggregates);
    drawDistributionCharts(run, aggregates);
  }

  const warnings = distributionWarnings(run, aggregates);
  elements.distWarnings.replaceChildren(...warnings.map(text => {
    const warning = document.createElement("span");
    warning.textContent = text;
    return warning;
  }));
  elements.distWarnings.hidden = !warnings.length;
}

function exportDistributionJson() {
  if (!activeDistributionRun?.probes.length) return;
  downloadEvaluation(`logit-scope-distribution-${activeDistributionRun.id}.json`, "application/json",
    JSON.stringify(activeDistributionRun, null, 2));
}

function exportDistributionCsv() {
  const run = activeDistributionRun;
  if (!run?.probes.length) return;
  const aggregates = new Map(distributionAggregates(run).map(aggregate => [aggregate.descriptor.key, aggregate]));
  const descriptors = distributionStageDescriptors(run);
  const headings = [
    "task_id", "task_kind", "run_id", "block", "rotation", "mapping_id", "stage", "configuration_id",
    "configuration_name", "model_path", "requested_assistant_prefix", "assistant_prefix", "label_token_prefix",
    "assistant_prefix_auto_selected", "auto_select_assistant_prefix", "formatted_prompt", "prompt_token_count",
    "sample_count", "seed", "profile", "diversity", "candidate_cap", "minimum_relative_probability", "protect_control_tokens",
    "outcome_id", "outcome_text", "label", "token_id", "target_probability", "probability", "projected_count",
    "retained", "rank", "valid_mass", "invalid_mass", "open_tv", "conditional_tv", "open_js_distance",
    "conditional_entropy_error", "missing_target_mass", "pairwise_order_accuracy", "support_size",
    "sampler_target_entropy", "sampler_shaped_entropy", "target_saturated", "pooled_open_tv",
    "pooled_conditional_tv", "mapping_sensitivity"
  ];
  const rows = [];
  for (const probe of run.probes) {
    for (const descriptor of descriptors) {
      const stage = descriptor.stage(probe);
      if (!stage) continue;
      const configuration = descriptor.configurationId ?
        probe.result.configurations.find(value => value.id === descriptor.configurationId) : null;
      const aggregate = aggregates.get(descriptor.key);
      for (const outcome of stage.outcomes) {
        rows.push([
          run.task.id, run.task.kind, run.id, probe.mapping.block, probe.mapping.rotation, probe.result.mappingId,
          descriptor.key, descriptor.configurationId || "", configuration?.name || "", probe.result.modelPath,
          probe.result.requestedAssistantPrefix ?? run.assistantPrefix, probe.result.assistantPrefix,
          probe.result.labelTokenPrefix || "", probe.result.assistantPrefixAutoSelected, run.autoSelectAssistantPrefix,
          probe.result.formattedPrompt,
          probe.result.promptTokenCount, probe.result.sampleCount, run.seed, configuration?.settings.profile,
          configuration?.settings.diversity,
          configuration?.settings.candidateCap, configuration?.settings.minimumRelativeProbability,
          configuration?.settings.protectControlTokens, outcome.id, outcome.text, outcome.label, outcome.tokenId,
          outcome.targetProbability, outcome.probability, outcome.projectedCount, outcome.retained, outcome.rank,
          stage.openMetrics.validMass, stage.openMetrics.invalidMass, stage.openMetrics.totalVariation,
          stage.conditionalMetrics.valid ? stage.conditionalMetrics.totalVariation : null,
          stage.openMetrics.jsDistanceNormalized, stage.conditionalMetrics.valid ?
            stage.conditionalMetrics.conditionalEntropyError : null, stage.openMetrics.missingTargetMass,
          stage.conditionalMetrics.valid ? stage.conditionalMetrics.pairwiseOrderAccuracy : null,
          configuration?.diagnostics.supportSize, configuration?.diagnostics.samplerTargetEntropy,
          configuration?.diagnostics.samplerShapedEntropy, configuration?.diagnostics.targetSaturated,
          aggregate?.openTv, aggregate?.conditionalTv, aggregate?.mappingSensitivity
        ]);
      }
    }
  }
  const csv = [headings, ...rows].map(row => row.map(csvCell).join(",")).join("\r\n");
  downloadEvaluation(`logit-scope-distribution-${run.id}.csv`, "text/csv;charset=utf-8", csv);
}

function restoreDistributionRunControls(run, restoreConfigurations) {
  elements.distTask.value = run.task.id;
  elements.distBlocks.value = run.blocks;
  elements.distDraws.value = run.sampleCount;
  elements.distSeed.value = run.seed;
  if (!restoreConfigurations) return;
  elements.distPrefix.value = run.assistantPrefix;
  elements.distAutoPrefix.checked = run.autoSelectAssistantPrefix !== false;
  elements.distLockSupport.checked = run.supportLocked;
  for (const side of ["A", "B"]) {
    const configuration = run.configurations.find(value => value.id === side);
    if (!configuration) continue;
    distributionField(side, "Profile").value = configuration.settings.profile;
    distributionField(side, "Diversity").value = Math.round(configuration.settings.diversity * 100);
    distributionField(side, "Cap").value = String(configuration.settings.candidateCap);
    distributionField(side, "Floor").value = String(configuration.settings.minimumRelativeProbability);
    distributionField(side, "Guard").checked = configuration.settings.protectControlTokens;
  }
}

function initializeDistributionLab() {
  for (const task of distributionTasks) {
    const option = document.createElement("option");
    option.value = task.id;
    option.textContent = `${task.kind === "control" ? "Controls" : "Implicit"} · ${task.name}`;
    elements.distTask.append(option);
    const mappings = createBalancedDistributionMappings(normalizedTaskOutcomes(task), 2, 1234);
    verifyBalancedDistributionBlock(mappings, normalizedTaskOutcomes(task), 0);
    verifyBalancedDistributionBlock(mappings, normalizedTaskOutcomes(task), 1);
  }
  window.logitScopeDistributionMappings = {
    create: createBalancedDistributionMappings,
    verifyBlock: verifyBalancedDistributionBlock
  };
  if (activeDistributionRun?.task?.id) {
    const resumable = ["paused", "stopped"].includes(activeDistributionRun.status) &&
      activeDistributionRun.nextMappingIndex < activeDistributionRun.mappings.length;
    restoreDistributionRunControls(activeDistributionRun, resumable);
  }
  renderDistributionTask();
  synchronizeDistributionSupport();
  saveDistributionRun();
  renderDistributionResults();
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
elements.distTask.addEventListener("change", renderDistributionTask);
for (const side of ["A", "B"])
  distributionField(side, "Profile").addEventListener("change", updateDistributionConfigControls);
for (const name of ["Cap", "Floor", "Guard"])
  distributionField("A", name).addEventListener("change", synchronizeDistributionSupport);
elements.distLockSupport.addEventListener("change", synchronizeDistributionSupport);
elements.distStart.addEventListener("click", startDistributionRun);
elements.distStop.addEventListener("click", stopDistributionRun);
elements.distClear.addEventListener("click", clearDistributionRun);
elements.distExportJson.addEventListener("click", exportDistributionJson);
elements.distExportCsv.addEventListener("click", exportDistributionCsv);
window.addEventListener("resize", () => {
  drawPlot();
  if (activeDistributionRun?.probes.length)
    drawDistributionCharts(activeDistributionRun, distributionAggregates(activeDistributionRun));
});

renderEvaluationSelector();
renderEvaluationSummary();
showNextUnjudgedTrial();
initializeDistributionLab();
updateControlLabels();
drawPlot();
poll();
