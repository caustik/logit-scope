#!/usr/bin/env python3
"""Collect and analyze Distribution Lab probes without target leakage.

The browser lab is useful for inspecting one task, but sampler research needs a
repeatable corpus and held-out evaluation. This script reproduces the browser's
built-in tasks and balanced mappings, collects exact next-token probabilities
through the local API, and compares:

* raw, Temperature, and Soliton label-conditional distributions;
* permutation ensembles that average semantic probabilities across mappings;
* target-free transferable label/position debiasing; and
* supervised scalar and affine calibration with leave-one-task-out fitting; and
* direct semantic-probability inference.

Only training tasks contribute target probabilities to a held-out calibrator.
Oracle temperature is reported solely as a diagnostic lower bound for monotonic
reshaping and must not be treated as a deployable result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict


TASKS = [
    {
        "id": "two-dice-sum",
        "name": "Sum of two fair six-sided dice",
        "kind": "implicit",
        "description": "Two independent fair six-sided dice are rolled. The outcome is their sum.",
        "outcomes": [
            {"id": f"sum-{index + 2}", "text": f"sum {index + 2}", "weight": weight}
            for index, weight in enumerate([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
        ],
    },
    {
        "id": "four-coin-heads",
        "name": "Number of heads in four fair tosses",
        "kind": "implicit",
        "description": "A fair coin is tossed independently four times. The outcome is the total number of heads.",
        "outcomes": [
            {"id": f"heads-{heads}", "text": f"{heads} {'head' if heads == 1 else 'heads'}", "weight": weight}
            for heads, weight in enumerate([1, 4, 6, 4, 1])
        ],
    },
    {
        "id": "two-dice-maximum",
        "name": "Maximum of two fair six-sided dice",
        "kind": "implicit",
        "description": "Two independent fair six-sided dice are rolled. The outcome is the larger of the two values.",
        "outcomes": [
            {"id": f"maximum-{index + 1}", "text": f"maximum {index + 1}", "weight": weight}
            for index, weight in enumerate([1, 3, 5, 7, 9, 11])
        ],
    },
    {
        "id": "first-head-three-tosses",
        "name": "First head position in three fair tosses",
        "kind": "implicit",
        "description": (
            "A fair coin is tossed up to three times. The outcome is the toss on which the first head appears, or no heads."
        ),
        "outcomes": [
            {"id": "head-1", "text": "head on toss 1", "weight": 4},
            {"id": "head-2", "text": "head on toss 2", "weight": 2},
            {"id": "head-3", "text": "head on toss 3", "weight": 1},
            {"id": "no-heads", "text": "no heads", "weight": 1},
        ],
    },
    {
        "id": "explicit-five-way-control",
        "name": "Explicit 8:4:2:1:1 control",
        "kind": "control",
        "description": (
            "A calibrated source emits outcome one, two, three, four, or five with relative frequencies "
            "8, 4, 2, 1, and 1 respectively."
        ),
        "outcomes": [
            {"id": f"control-{index + 1}", "text": f"outcome {index + 1}", "weight": weight}
            for index, weight in enumerate([8, 4, 2, 1, 1])
        ],
    },
]

MASK_32 = (1 << 32) - 1
MIN_PROBABILITY = 1e-300


def normalized_task(task):
    total = sum(outcome["weight"] for outcome in task["outcomes"])
    return {
        **task,
        "outcomes": [{**outcome, "targetProbability": outcome["weight"] / total} for outcome in task["outcomes"]],
    }


def mulberry32(seed):
    state = seed & MASK_32

    def random():
        nonlocal state
        state = (state + 0x6D2B79F5) & MASK_32
        value = state
        value = ((value ^ (value >> 15)) * (value | 1)) & MASK_32
        value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & MASK_32)) & MASK_32
        value &= MASK_32
        return ((value ^ (value >> 14)) & MASK_32) / 4294967296

    return random


def shuffled(values, random):
    result = list(values)
    for index in range(len(result) - 1, 0, -1):
        other = math.floor(random() * (index + 1))
        result[index], result[other] = result[other], result[index]
    return result


def coprime_stride(size, random):
    if size <= 2:
        return 1
    candidates = [stride for stride in range(1, size) if math.gcd(stride, size) == 1]
    return candidates[math.floor(random() * len(candidates))]


def create_balanced_mappings(outcomes, block_count, seed):
    labels = [chr(ord("A") + index) for index in range(len(outcomes))]
    mappings = []
    size = len(outcomes)
    for block in range(block_count):
        random = mulberry32(seed + block)
        base_outcomes = shuffled(outcomes, random)
        base_labels = shuffled(labels, random)
        stride = coprime_stride(size, random)
        for rotation in range(size):
            assigned_labels = {
                outcome["id"]: base_labels[(index + rotation) % size] for index, outcome in enumerate(outcomes)
            }
            display_offset = (rotation * stride) % size
            rows = []
            for position in range(size):
                outcome = base_outcomes[(position + display_offset) % size]
                rows.append({"position": position, "outcomeId": outcome["id"], "label": assigned_labels[outcome["id"]]})
            mappings.append(
                {"id": f"{block}:{rotation}", "block": block, "rotation": rotation, "stride": stride, "rows": rows}
            )
    return mappings


def verify_balanced_mappings(mappings, outcomes, block_count):
    size = len(outcomes)
    expected_labels = {chr(ord("A") + index) for index in range(size)}
    expected_positions = set(range(size))
    for block in range(block_count):
        block_mappings = [mapping for mapping in mappings if mapping["block"] == block]
        if len(block_mappings) != size:
            raise ValueError(f"Block {block} has {len(block_mappings)} mappings, expected {size}")
        for outcome in outcomes:
            rows = [
                next(row for row in mapping["rows"] if row["outcomeId"] == outcome["id"]) for mapping in block_mappings
            ]
            if {row["label"] for row in rows} != expected_labels:
                raise ValueError(f"{outcome['id']} did not receive every label in block {block}")
            if {row["position"] for row in rows} != expected_positions:
                raise ValueError(f"{outcome['id']} did not occupy every position in block {block}")


def build_prompt(task, mapping):
    outcomes = {outcome["id"]: outcome for outcome in task["outcomes"]}
    mapping_lines = "\n".join(f"{row['label']} = {outcomes[row['outcomeId']]['text']}" for row in mapping["rows"])
    return f"""\
{task["description"]}

The possible outcomes are represented by opaque labels:
{mapping_lines}

Simulate one independent occurrence of this process and return the label for its outcome. Sample according to the true probability distribution implied by the process. Do not merely choose the most likely outcome. Do not explain, show calculations, list probabilities, or add punctuation.

Return exactly one label."""


def api_request(base_url, path, method="GET", body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} returned HTTP {error.code}: {detail}") from error


def collect_probe(base_url, task, mapping, assistant_prefix, seed, timeout):
    labels = {row["outcomeId"]: row["label"] for row in mapping["rows"]}
    body = {
        "prompt": build_prompt(task, mapping),
        "assistantPrefix": assistant_prefix,
        "autoSelectAssistantPrefix": True,
        "mappingId": f"{task['id']}:block-{mapping['block']}:rotation-{mapping['rotation']}",
        "sampleCount": 0,
        "seed": seed,
        "outcomes": [
            {
                "id": outcome["id"],
                "text": outcome["text"],
                "label": labels[outcome["id"]],
                "targetProbability": outcome["targetProbability"],
            }
            for outcome in task["outcomes"]
        ],
        "configurations": [
            {
                "id": "temperature",
                "name": "Temperature",
                "settings": {
                    "profile": "temperature",
                    "diversity": 1.88,
                    "candidateCap": 4096,
                    "minimumRelativeProbability": 0,
                    "seed": seed,
                    "protectControlTokens": False,
                },
            },
            {
                "id": "soliton",
                "name": "Soliton",
                "settings": {
                    "profile": "soliton",
                    "diversity": 1.88,
                    "candidateCap": 4096,
                    "minimumRelativeProbability": 0,
                    "seed": seed,
                    "protectControlTokens": False,
                },
            },
        ],
    }
    accepted = api_request(base_url, "/api/distribution", "POST", body)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = api_request(base_url, "/api/distribution")
        if result.get("id") == accepted["id"] and result.get("ready"):
            if result.get("status") != "Complete":
                raise RuntimeError(f"{body['mappingId']}: {result.get('status', 'unknown failure')}")
            return {"taskId": task["id"], "mapping": mapping, "result": result}
        time.sleep(0.085)
    raise TimeoutError(f"{body['mappingId']} timed out after {timeout:g} seconds")


def collect_dataset(args):
    snapshot = api_request(args.url, "/api/snapshot")
    if not snapshot.get("modelLoaded") or snapshot.get("generating"):
        raise RuntimeError(f"Model is not ready: {snapshot.get('status', 'unknown status')}")
    selected_ids = set(args.tasks.split(",")) if args.tasks != "all" else {task["id"] for task in TASKS}
    unknown_ids = selected_ids - {task["id"] for task in TASKS}
    if unknown_ids:
        raise ValueError(f"Unknown task IDs: {', '.join(sorted(unknown_ids))}")
    tasks = [normalized_task(task) for task in TASKS if task["id"] in selected_ids]
    probes = []
    total_probes = sum(len(task["outcomes"]) * args.blocks for task in tasks)
    for task in tasks:
        mappings = create_balanced_mappings(task["outcomes"], args.blocks, args.seed)
        verify_balanced_mappings(mappings, task["outcomes"], args.blocks)
        for mapping in mappings:
            print(
                f"[{len(probes) + 1:>2}/{total_probes}] {task['id']} "
                f"block {mapping['block'] + 1}, rotation {mapping['rotation'] + 1}",
                flush=True,
            )
            probes.append(collect_probe(args.url, task, mapping, args.assistant_prefix, args.seed, args.timeout))
    return {
        "version": 1,
        "collectedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseUrl": args.url,
        "seed": args.seed,
        "blocks": args.blocks,
        "assistantPrefix": args.assistant_prefix,
        "modelPath": probes[0]["result"]["modelPath"] if probes else "",
        "tasks": tasks,
        "probes": probes,
    }


def parse_probability_object(response, task):
    candidates = re.findall(r"\{.*?\}", response, flags=re.DOTALL)
    expected_ids = {outcome["id"] for outcome in task["outcomes"]}
    for candidate in reversed(candidates):
        def replace_fraction(match):
            denominator = float(match.group(3))
            return f"{match.group(1)}{float(match.group(2)) / denominator:.17g}"

        json_candidate = re.sub(
            r"(:\s*)(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", replace_fraction, candidate
        )
        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or set(parsed) != expected_ids:
            continue
        try:
            values = [float(parsed[outcome["id"]]) for outcome in task["outcomes"]]
        except (TypeError, ValueError):
            continue
        if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
            continue
        return normalize(values)
    marker_values = []
    for outcome in task["outcomes"]:
        key = re.escape(outcome["id"])
        match = re.search(
            rf"(?im)^\s*PROB\s+{key}\s*=\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
            rf"(?:\s*/\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?\s*$",
            response,
        )
        if not match:
            break
        numerator = float(match.group(1))
        denominator = float(match.group(2)) if match.group(2) else 1.0
        if not math.isfinite(numerator) or not math.isfinite(denominator) or numerator < 0 or denominator <= 0:
            break
        marker_values.append(numerator / denominator)
    if len(marker_values) == len(task["outcomes"]):
        return normalize(marker_values)
    raise ValueError("Response did not contain one JSON object with exactly the requested outcome IDs")


def collect_probability_inference(base_url, task, seed, timeout, thinking):
    outcome_lines = "\n".join(f"{outcome['id']} = {outcome['text']}" for outcome in task["outcomes"])
    if thinking:
        output_instruction = (
            "Reason through the process concisely. Immediately after computing each outcome, write a standalone line "
            "in the exact form `PROB outcome-id = decimal-or-single-fraction`. Emit every PROB line before any final "
            "explanation. Then return one compact JSON object with the same probabilities."
        )
        thinking_switch = ""
        assistant_prefix = (
            "<think>\nI will calculate every requested probability exactly, keep the reasoning under 350 tokens, "
            "then close the thinking block and output the requested JSON immediately.\n"
        )
    else:
        output_instruction = (
            "Return only one compact JSON object mapping each outcome ID to its probability as a decimal number or a "
            "single numeric fraction."
        )
        thinking_switch = "\n/no_think"
        assistant_prefix = ""
    prompt = f"""\
{task["description"]}

The possible outcomes are:
{outcome_lines}

Infer the exact probability of every outcome. {output_instruction} The probabilities must sum to 1. Do not sample an outcome.{thinking_switch}"""
    body = {
        "prompt": prompt,
        "assistantPrefix": assistant_prefix,
        "settings": {
            "profile": "temperature",
            "diversity": 0.6 if thinking else 0.7,
            "candidateCap": 64,
            "minimumRelativeProbability": 0,
            "seed": seed,
            "protectControlTokens": True,
        },
    }
    accepted = api_request(base_url, "/api/evaluation", "POST", body)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = api_request(base_url, "/api/evaluation")
        if result.get("id") == accepted["id"] and result.get("ready"):
            probabilities = parse_probability_object(result["response"], task)
            return {
                "taskId": task["id"],
                "thinking": thinking,
                "generationStatus": result.get("status", ""),
                "prompt": prompt,
                "response": result["response"],
                "tokenCount": result["tokenCount"],
                "probabilities": probabilities,
            }
        time.sleep(0.085)
    raise TimeoutError(f"{task['id']} probability inference timed out after {timeout:g} seconds")


def add_probability_inferences(dataset, args):
    inferences = []
    for index, task in enumerate(dataset["tasks"]):
        print(f"[{index + 1}/{len(dataset['tasks'])}] infer {task['id']}", flush=True)
        try:
            inference = collect_probability_inference(
                args.url, task, args.seed + index, args.timeout, args.inference_thinking
            )
            inference["status"] = "complete"
            inferences.append(inference)
        except Exception as error:
            inferences.append({"taskId": task["id"], "status": "error", "error": str(error)})
            print(f"  failed: {error}", flush=True)
    return {**dataset, "probabilityInferences": inferences}


def create_procedural_tasks(count, seed):
    source = random.Random(seed)
    tasks = []
    probability_choices = [(1, 4), (1, 3), (2, 5), (3, 5), (2, 3), (3, 4)]
    for task_index in range(count):
        family = task_index % 5
        suffix = f"{task_index + 1:03d}"
        if family == 0:
            numerator, denominator = source.choice(probability_choices)
            trials = source.randint(3, 7)
            weights = [
                math.comb(trials, successes)
                * numerator**successes
                * (denominator - numerator) ** (trials - successes)
                for successes in range(trials + 1)
            ]
            tasks.append(
                {
                    "id": f"procedural-binomial-{suffix}",
                    "name": f"Binomial n={trials}, p={numerator}/{denominator}",
                    "kind": "procedural",
                    "family": "binomial",
                    "description": (
                        f"A trial succeeds with probability {numerator}/{denominator} and fails otherwise. "
                        f"The trial is repeated independently {trials} times. The outcome is the total number of successes."
                    ),
                    "outcomes": [
                        {"id": f"successes-{successes}", "text": f"{successes} successes", "weight": weight}
                        for successes, weight in enumerate(weights)
                    ],
                }
            )
        elif family == 1:
            numerator, denominator = source.choice(probability_choices)
            attempts = source.randint(3, 7)
            failure = denominator - numerator
            outcomes = [
                {
                    "id": f"first-{attempt}",
                    "text": f"first success on attempt {attempt}",
                    "weight": numerator * failure ** (attempt - 1) * denominator ** (attempts - attempt),
                }
                for attempt in range(1, attempts + 1)
            ]
            outcomes.append(
                {"id": "no-success", "text": "no success", "weight": failure**attempts}
            )
            tasks.append(
                {
                    "id": f"procedural-first-success-{suffix}",
                    "name": f"First success n={attempts}, p={numerator}/{denominator}",
                    "kind": "procedural",
                    "family": "first-success",
                    "description": (
                        f"Each independent attempt succeeds with probability {numerator}/{denominator}. "
                        f"At most {attempts} attempts are made. The outcome is the attempt of the first success, or no success."
                    ),
                    "outcomes": outcomes,
                }
            )
        elif family == 2:
            left_sides = source.randint(3, 8)
            right_sides = source.randint(3, 8)
            weights = []
            for total in range(2, left_sides + right_sides + 1):
                weights.append(
                    sum(
                        1
                        for left in range(1, left_sides + 1)
                        for right in range(1, right_sides + 1)
                        if left + right == total
                    )
                )
            tasks.append(
                {
                    "id": f"procedural-dice-sum-{suffix}",
                    "name": f"Sum of d{left_sides} and d{right_sides}",
                    "kind": "procedural",
                    "family": "dice-sum",
                    "description": (
                        f"One fair {left_sides}-sided die and one fair {right_sides}-sided die are rolled independently. "
                        "Both dice are numbered consecutively starting at 1. The outcome is their sum."
                    ),
                    "outcomes": [
                        {"id": f"sum-{total}", "text": f"sum {total}", "weight": weight}
                        for total, weight in zip(range(2, left_sides + right_sides + 1), weights)
                    ],
                }
            )
        elif family == 3:
            left_sides = source.randint(3, 8)
            right_sides = source.randint(3, 8)
            maximum_value = max(left_sides, right_sides)
            weights = []
            previous_count = 0
            for value in range(1, maximum_value + 1):
                cumulative_count = min(value, left_sides) * min(value, right_sides)
                weights.append(cumulative_count - previous_count)
                previous_count = cumulative_count
            tasks.append(
                {
                    "id": f"procedural-dice-maximum-{suffix}",
                    "name": f"Maximum of d{left_sides} and d{right_sides}",
                    "kind": "procedural",
                    "family": "dice-maximum",
                    "description": (
                        f"One fair {left_sides}-sided die and one fair {right_sides}-sided die are rolled independently. "
                        "Both dice are numbered consecutively starting at 1. The outcome is the larger value."
                    ),
                    "outcomes": [
                        {"id": f"maximum-{value}", "text": f"maximum {value}", "weight": weight}
                        for value, weight in enumerate(weights, start=1)
                    ],
                }
            )
        else:
            red = source.randint(3, 8)
            blue = source.randint(3, 8)
            draws = source.randint(2, min(5, red + blue))
            minimum_red = max(0, draws - blue)
            maximum_red = min(draws, red)
            tasks.append(
                {
                    "id": f"procedural-urn-{suffix}",
                    "name": f"Urn {red} red/{blue} blue, draw {draws}",
                    "kind": "procedural",
                    "family": "hypergeometric",
                    "description": (
                        f"An urn contains {red} red balls and {blue} blue balls. "
                        f"{draws} balls are drawn uniformly without replacement. "
                        "The outcome is the number of red balls drawn."
                    ),
                    "outcomes": [
                        {
                            "id": f"red-{red_drawn}",
                            "text": f"{red_drawn} red balls",
                            "weight": math.comb(red, red_drawn) * math.comb(blue, draws - red_drawn),
                        }
                        for red_drawn in range(minimum_red, maximum_red + 1)
                    ],
                }
            )
    return [normalized_task(task) for task in tasks]


def create_procedural_dataset(args):
    return {
        "version": 1,
        "collectedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseUrl": args.url,
        "seed": args.seed,
        "blocks": 0,
        "assistantPrefix": "",
        "modelPath": "",
        "tasks": create_procedural_tasks(args.procedural_count, args.seed),
        "probes": [],
    }


def normalize(values):
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return [value / total for value in values]


def softmax(logits):
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    return normalize(weights)


def total_variation(left, right):
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def stage_for_probe(probe, stage_name):
    if stage_name == "fullRaw":
        return probe["result"]["fullRaw"]
    configuration_id, stage_key = stage_name.split(".", 1)
    configuration = next(
        configuration for configuration in probe["result"]["configurations"] if configuration["id"] == configuration_id
    )
    return configuration[stage_key]


def task_lookup(dataset):
    return {task["id"]: task for task in dataset["tasks"]}


def ordered_stage_probabilities(probe, task, stage_name="fullRaw"):
    outcomes = {outcome["id"]: outcome for outcome in stage_for_probe(probe, stage_name)["outcomes"]}
    return [outcomes[outcome["id"]]["probability"] for outcome in task["outcomes"]]


def conditional_probabilities(probe, task, stage_name="fullRaw"):
    return normalize(ordered_stage_probabilities(probe, task, stage_name))


def target_probabilities(task):
    return [outcome["targetProbability"] for outcome in task["outcomes"]]


def mapping_features(probe, task):
    rows = {row["outcomeId"]: row for row in probe["mapping"]["rows"]}
    labels = [ord(rows[outcome["id"]]["label"]) - ord("A") for outcome in task["outcomes"]]
    positions = [rows[outcome["id"]]["position"] for outcome in task["outcomes"]]
    return labels, positions


def task_probes(dataset):
    grouped = defaultdict(list)
    for probe in dataset["probes"]:
        grouped[probe["taskId"]].append(probe)
    return grouped


def mean(values):
    if not isinstance(values, (list, tuple)):
        values = list(values)
    return sum(values) / len(values) if values else math.nan


def pooled_vector(probes, task, stage_name="fullRaw"):
    vectors = [conditional_probabilities(probe, task, stage_name) for probe in probes]
    return [mean(values) for values in zip(*vectors)]


def geometric_pooled_vector(probes, task, stage_name="fullRaw"):
    vectors = [conditional_probabilities(probe, task, stage_name) for probe in probes]
    return softmax(
        [mean(math.log(max(MIN_PROBABILITY, vector[index])) for vector in vectors) for index in range(len(vectors[0]))]
    )


def fit_affine_calibrator(examples, maximum_size, use_label_bias, use_position_bias, steps=3000):
    parameter_count = 1 + 2 * maximum_size
    parameters = [0.0] * parameter_count
    if not examples:
        return {
            "scale": 1.0,
            "labelBias": [0.0] * maximum_size,
            "positionBias": [0.0] * maximum_size,
        }
    first_moment = [0.0] * parameter_count
    second_moment = [0.0] * parameter_count
    observed_labels = {label for example in examples for label in example["labels"]}
    observed_positions = {position for example in examples for position in example["positions"]}
    task_counts = defaultdict(int)
    for example in examples:
        task_counts[example["taskId"]] += 1
    task_weight = 1.0 / len(task_counts)

    for step in range(1, steps + 1):
        gradients = [0.0] * parameter_count
        scale = math.exp(parameters[0])
        for example in examples:
            weight = task_weight / task_counts[example["taskId"]]
            logits = []
            for log_probability, label, position in zip(
                example["logProbabilities"], example["labels"], example["positions"]
            ):
                value = scale * log_probability
                if use_label_bias:
                    value += parameters[1 + label]
                if use_position_bias:
                    value += parameters[1 + maximum_size + position]
                logits.append(value)
            predicted = softmax(logits)
            differences = [
                weight * (prediction - target) for prediction, target in zip(predicted, example["target"])
            ]
            gradients[0] += sum(
                difference * scale * log_probability
                for difference, log_probability in zip(differences, example["logProbabilities"])
            )
            if use_label_bias:
                for difference, label in zip(differences, example["labels"]):
                    gradients[1 + label] += difference
            if use_position_bias:
                for difference, position in zip(differences, example["positions"]):
                    gradients[1 + maximum_size + position] += difference

        regularization = 1e-3
        gradients[0] += regularization * parameters[0]
        for index in range(1, parameter_count):
            gradients[index] += regularization * parameters[index]
        learning_rate = 0.03
        for index, gradient in enumerate(gradients):
            first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient
            second_moment[index] = 0.999 * second_moment[index] + 0.001 * gradient * gradient
            corrected_first = first_moment[index] / (1 - 0.9**step)
            corrected_second = second_moment[index] / (1 - 0.999**step)
            parameters[index] -= learning_rate * corrected_first / (math.sqrt(corrected_second) + 1e-8)
        parameters[0] = min(3.0, max(-3.0, parameters[0]))
        if use_label_bias and observed_labels:
            label_mean = mean(parameters[1 + index] for index in observed_labels)
            for index in observed_labels:
                parameters[1 + index] -= label_mean
        if use_position_bias and observed_positions:
            position_mean = mean(parameters[1 + maximum_size + index] for index in observed_positions)
            for index in observed_positions:
                parameters[1 + maximum_size + index] -= position_mean
    return {
        "scale": math.exp(parameters[0]),
        "labelBias": parameters[1 : 1 + maximum_size] if use_label_bias else [0.0] * maximum_size,
        "positionBias": parameters[1 + maximum_size :] if use_position_bias else [0.0] * maximum_size,
    }


def calibration_examples(dataset, excluded_task_id=None):
    tasks = task_lookup(dataset)
    examples = []
    for probe in dataset["probes"]:
        if probe["taskId"] == excluded_task_id:
            continue
        task = tasks[probe["taskId"]]
        probabilities = conditional_probabilities(probe, task)
        labels, positions = mapping_features(probe, task)
        examples.append(
            {
                "taskId": task["id"],
                "logProbabilities": [math.log(max(MIN_PROBABILITY, value)) for value in probabilities],
                "target": target_probabilities(task),
                "labels": labels,
                "positions": positions,
            }
        )
    return examples


def pooled_calibration_examples(dataset, excluded_task_id=None):
    tasks = task_lookup(dataset)
    grouped = task_probes(dataset)
    examples = []
    for task_id, task in tasks.items():
        if task_id == excluded_task_id:
            continue
        probabilities = pooled_vector(grouped[task_id], task)
        examples.append(
            {
                "taskId": task_id,
                "logProbabilities": [math.log(max(MIN_PROBABILITY, value)) for value in probabilities],
                "target": target_probabilities(task),
                "labels": [0] * len(probabilities),
                "positions": [0] * len(probabilities),
            }
        )
    return examples


def apply_calibrator(probabilities, labels, positions, calibrator):
    logits = []
    for probability, label, position in zip(probabilities, labels, positions):
        logits.append(
            calibrator["scale"] * math.log(max(MIN_PROBABILITY, probability))
            + calibrator["labelBias"][label]
            + calibrator["positionBias"][position]
        )
    return softmax(logits)


def fit_transferable_bias(dataset, excluded_task_id, maximum_size):
    tasks = task_lookup(dataset)
    observations = []
    for probe in dataset["probes"]:
        if probe["taskId"] == excluded_task_id:
            continue
        task = tasks[probe["taskId"]]
        probabilities = conditional_probabilities(probe, task)
        labels, positions = mapping_features(probe, task)
        for index, outcome in enumerate(task["outcomes"]):
            observations.append(
                {
                    "semantic": (task["id"], outcome["id"]),
                    "label": labels[index],
                    "position": positions[index],
                    "value": math.log(max(MIN_PROBABILITY, probabilities[index])),
                }
            )
    if not observations:
        return {"label": [0.0] * maximum_size, "position": [0.0] * maximum_size}
    semantic_effects = defaultdict(float)
    label_effects = [0.0] * maximum_size
    position_effects = [0.0] * maximum_size
    observed_labels = {observation["label"] for observation in observations}
    observed_positions = {observation["position"] for observation in observations}
    for _ in range(100):
        semantic_values = defaultdict(list)
        for observation in observations:
            semantic_values[observation["semantic"]].append(
                observation["value"]
                - label_effects[observation["label"]]
                - position_effects[observation["position"]]
            )
        semantic_effects = {key: mean(values) for key, values in semantic_values.items()}
        label_values = defaultdict(list)
        for observation in observations:
            label_values[observation["label"]].append(
                observation["value"]
                - semantic_effects[observation["semantic"]]
                - position_effects[observation["position"]]
            )
        for index, values in label_values.items():
            label_effects[index] = mean(values)
        label_mean = mean(label_effects[index] for index in observed_labels)
        for index in observed_labels:
            label_effects[index] -= label_mean
        position_values = defaultdict(list)
        for observation in observations:
            position_values[observation["position"]].append(
                observation["value"] - semantic_effects[observation["semantic"]] - label_effects[observation["label"]]
            )
        for index, values in position_values.items():
            position_effects[index] = mean(values)
        position_mean = mean(position_effects[index] for index in observed_positions)
        for index in observed_positions:
            position_effects[index] -= position_mean
    return {"label": label_effects, "position": position_effects}


def apply_transferable_bias(probabilities, labels, positions, bias):
    return softmax(
        [
            math.log(max(MIN_PROBABILITY, probability)) - bias["label"][label] - bias["position"][position]
            for probability, label, position in zip(probabilities, labels, positions)
        ]
    )


def fit_oracle_scale(probabilities, target):
    best_scale = 1.0
    best_tv = total_variation(probabilities, target)
    for index in range(2001):
        log_scale = -4.0 + 8.0 * index / 2000
        scale = math.exp(log_scale)
        candidate = softmax([scale * math.log(max(MIN_PROBABILITY, value)) for value in probabilities])
        candidate_tv = total_variation(candidate, target)
        if candidate_tv < best_tv:
            best_tv = candidate_tv
            best_scale = scale
    return best_scale, best_tv


def analyze_dataset(dataset):
    tasks = task_lookup(dataset)
    grouped = task_probes(dataset)
    maximum_size = max(len(task["outcomes"]) for task in tasks.values())
    report = {"modelPath": dataset.get("modelPath", ""), "tasks": {}, "macro": {}}
    cross_validated = defaultdict(list)
    probability_inferences = {
        inference["taskId"]: inference
        for inference in dataset.get("probabilityInferences", [])
        if inference.get("status", "complete") == "complete"
    }

    for held_out_id, held_out_task in tasks.items():
        held_out_probes = grouped[held_out_id]
        target = target_probabilities(held_out_task)
        scalar = fit_affine_calibrator(
            calibration_examples(dataset, held_out_id), maximum_size, False, False
        )
        affine = fit_affine_calibrator(
            calibration_examples(dataset, held_out_id), maximum_size, True, True
        )
        pooled_scalar = fit_affine_calibrator(
            pooled_calibration_examples(dataset, held_out_id), maximum_size, False, False
        )
        transferable_bias = fit_transferable_bias(dataset, held_out_id, maximum_size)
        raw_tvs = []
        scalar_tvs = []
        affine_tvs = []
        transferable_tvs = []
        scalar_vectors = []
        affine_vectors = []
        transferable_vectors = []
        for probe in held_out_probes:
            raw = conditional_probabilities(probe, held_out_task)
            labels, positions = mapping_features(probe, held_out_task)
            raw_tvs.append(total_variation(raw, target))
            scalar_vector = apply_calibrator(raw, labels, positions, scalar)
            affine_vector = apply_calibrator(raw, labels, positions, affine)
            transferable_vector = apply_transferable_bias(raw, labels, positions, transferable_bias)
            scalar_vectors.append(scalar_vector)
            affine_vectors.append(affine_vector)
            transferable_vectors.append(transferable_vector)
            scalar_tvs.append(total_variation(scalar_vector, target))
            affine_tvs.append(total_variation(affine_vector, target))
            transferable_tvs.append(total_variation(transferable_vector, target))
        held_out_pooled = pooled_vector(held_out_probes, held_out_task)
        directly_calibrated_pooled = apply_calibrator(
            held_out_pooled, [0] * len(held_out_pooled), [0] * len(held_out_pooled), pooled_scalar
        )
        cross_validated[held_out_id] = {
            "rawMeanMappingTv": mean(raw_tvs),
            "temperatureCalibrationMeanMappingTv": mean(scalar_tvs),
            "affineCalibrationMeanMappingTv": mean(affine_tvs),
            "targetFreeDebiasMeanMappingTv": mean(transferable_tvs),
            "temperatureCalibrationPooledTv": total_variation(
                [mean(values) for values in zip(*scalar_vectors)], target
            ),
            "affineCalibrationPooledTv": total_variation(
                [mean(values) for values in zip(*affine_vectors)], target
            ),
            "targetFreeDebiasPooledTv": total_variation(
                [mean(values) for values in zip(*transferable_vectors)], target
            ),
            "directPooledTemperatureCalibrationTv": total_variation(directly_calibrated_pooled, target),
            "temperatureScale": scalar["scale"],
            "affineScale": affine["scale"],
            "directPooledTemperatureScale": pooled_scalar["scale"],
        }

    for task_id, task in tasks.items():
        probes = grouped[task_id]
        target = target_probabilities(task)
        raw_vectors = [conditional_probabilities(probe, task) for probe in probes]
        pooled = pooled_vector(probes, task)
        geometric_pooled = geometric_pooled_vector(probes, task)
        oracle_scale, oracle_tv = fit_oracle_scale(pooled, target)
        valid_masses = [probe["result"]["fullRaw"]["openMetrics"]["validMass"] for probe in probes]
        resolved_boundaries = sorted(
            {
                (probe["result"]["assistantPrefix"], probe["result"]["labelTokenPrefix"])
                for probe in probes
            }
        )
        task_report = {
            "probeCount": len(probes),
            "validMassMean": mean(valid_masses),
            "validMassMinimum": min(valid_masses),
            "rawMeanMappingTv": mean(total_variation(vector, target) for vector in raw_vectors),
            "rawPooledTv": total_variation(pooled, target),
            "rawGeometricPooledTv": total_variation(geometric_pooled, target),
            "rawMappingSensitivityTv": mean(total_variation(vector, pooled) for vector in raw_vectors),
            "temperatureMeanMappingTv": mean(
                total_variation(conditional_probabilities(probe, task, "temperature.shaped"), target)
                for probe in probes
            ),
            "solitonMeanMappingTv": mean(
                total_variation(conditional_probabilities(probe, task, "soliton.shaped"), target) for probe in probes
            ),
            "oraclePooledTemperatureTv": oracle_tv,
            "oraclePooledTemperatureScale": oracle_scale,
            "target": target,
            "rawPooled": pooled,
            "resolvedBoundaries": [
                {"assistantPrefix": assistant_prefix, "labelTokenPrefix": label_prefix}
                for assistant_prefix, label_prefix in resolved_boundaries
            ],
            "inferredProbabilityTv": (
                total_variation(probability_inferences[task_id]["probabilities"], target)
                if task_id in probability_inferences
                else None
            ),
            **cross_validated[task_id],
        }
        report["tasks"][task_id] = task_report

    macro_keys = [
        "rawMeanMappingTv",
        "rawPooledTv",
        "rawGeometricPooledTv",
        "rawMappingSensitivityTv",
        "temperatureMeanMappingTv",
        "solitonMeanMappingTv",
        "oraclePooledTemperatureTv",
        "temperatureCalibrationMeanMappingTv",
        "affineCalibrationMeanMappingTv",
        "targetFreeDebiasMeanMappingTv",
        "temperatureCalibrationPooledTv",
        "affineCalibrationPooledTv",
        "targetFreeDebiasPooledTv",
        "directPooledTemperatureCalibrationTv",
    ]
    report["macro"] = {
        key: mean(task_report[key] for task_report in report["tasks"].values()) for key in macro_keys
    }
    inferred_values = [
        task_report["inferredProbabilityTv"]
        for task_report in report["tasks"].values()
        if task_report["inferredProbabilityTv"] is not None
    ]
    report["macro"]["inferredProbabilityTv"] = mean(inferred_values) if inferred_values else None
    return report


def analyze_inference_suite(dataset):
    inferences = {inference["taskId"]: inference for inference in dataset.get("probabilityInferences", [])}
    report = {"modelPath": dataset.get("modelPath", ""), "tasks": {}, "macro": {}}
    completed_tvs = []
    family_tvs = defaultdict(list)
    for task in dataset["tasks"]:
        inference = inferences.get(task["id"], {})
        task_report = {
            "family": task.get("family", "built-in"),
            "name": task["name"],
            "target": target_probabilities(task),
            "status": inference.get("status", "missing"),
        }
        if inference.get("status") == "complete" and "probabilities" in inference:
            task_tv = total_variation(inference["probabilities"], target_probabilities(task))
            completed_tvs.append(task_tv)
            family_tvs[task.get("family", "built-in")].append(task_tv)
            task_report.update(
                {
                    "inferredProbabilityTv": task_tv,
                    "inferred": inference["probabilities"],
                    "directTokenCount": inference["tokenCount"],
                }
            )
        else:
            task_report["error"] = inference.get("error", "missing inference result")
        report["tasks"][task["id"]] = task_report
    report["macro"] = {
        "taskCount": len(dataset["tasks"]),
        "completedCount": len(completed_tvs),
        "validRate": len(completed_tvs) / len(dataset["tasks"]) if dataset["tasks"] else 0,
        "inferredProbabilityTv": mean(completed_tvs) if completed_tvs else None,
        "maximumTv": max(completed_tvs) if completed_tvs else None,
        "familyMeanTv": {family: mean(values) for family, values in sorted(family_tvs.items())},
    }
    return report


def print_inference_suite_report(report):
    print()
    print(
        "Task".ljust(34)
        + "Family".ljust(18)
        + "Direct TV".ljust(11)
        + "Status"
    )
    print("-" * 33 + " " + "-" * 17 + " " + "-" * 10 + " " + "-" * 12)
    for task_id, task in report["tasks"].items():
        print(
            task_id.ljust(34)
            + task["family"].ljust(18)
            + (
                f"{task['inferredProbabilityTv']:.6f}"
                if task["status"] == "complete"
                else "—"
            ).ljust(11)
            + task["status"]
        )
    macro = report["macro"]
    print()
    direct_summary = (
        f"Direct valid {macro['completedCount']}/{macro['taskCount']} ({macro['validRate']:.1%}) · "
        f"macro TV {macro['inferredProbabilityTv']:.6f} · maximum TV {macro['maximumTv']:.6f}"
        if macro["inferredProbabilityTv"] is not None
        else f"Direct valid 0/{macro['taskCount']}"
    )
    print(direct_summary)
    for family, family_tv in macro["familyMeanTv"].items():
        print(f"  {family}: {family_tv:.6f}")


def print_report(report):
    headings = [
        ("Task", 29),
        ("Raw/map", 9),
        ("Raw/pool", 9),
        ("Temp", 9),
        ("Soliton", 9),
        ("CV-temp", 9),
        ("CV-affine", 10),
        ("Pool+CV", 9),
        ("PoolDir", 9),
        ("Infer", 9),
        ("Oracle", 9),
        ("Valid", 9),
    ]
    print()
    print("".join(label.ljust(width) for label, width in headings))
    print("".join("-" * (width - 1) + " " for _, width in headings))
    for task_id, task in report["tasks"].items():
        values = [
            task_id,
            f"{task['rawMeanMappingTv']:.4f}",
            f"{task['rawPooledTv']:.4f}",
            f"{task['temperatureMeanMappingTv']:.4f}",
            f"{task['solitonMeanMappingTv']:.4f}",
            f"{task['temperatureCalibrationMeanMappingTv']:.4f}",
            f"{task['affineCalibrationMeanMappingTv']:.4f}",
            f"{task['temperatureCalibrationPooledTv']:.4f}",
            f"{task['directPooledTemperatureCalibrationTv']:.4f}",
            f"{task['inferredProbabilityTv']:.4f}" if task["inferredProbabilityTv"] is not None else "n/a",
            f"{task['oraclePooledTemperatureTv']:.4f}",
            f"{task['validMassMean']:.2%}",
        ]
        print("".join(value.ljust(width) for value, (_, width) in zip(values, headings)))
    macro = report["macro"]
    values = [
        "MACRO MEAN",
        f"{macro['rawMeanMappingTv']:.4f}",
        f"{macro['rawPooledTv']:.4f}",
        f"{macro['temperatureMeanMappingTv']:.4f}",
        f"{macro['solitonMeanMappingTv']:.4f}",
        f"{macro['temperatureCalibrationMeanMappingTv']:.4f}",
        f"{macro['affineCalibrationMeanMappingTv']:.4f}",
        f"{macro['temperatureCalibrationPooledTv']:.4f}",
        f"{macro['directPooledTemperatureCalibrationTv']:.4f}",
        f"{macro['inferredProbabilityTv']:.4f}" if macro["inferredProbabilityTv"] is not None else "n/a",
        f"{macro['oraclePooledTemperatureTv']:.4f}",
        "",
    ]
    print("".join(value.ljust(width) for value, (_, width) in zip(values, headings)))
    print()
    print("CV methods fit only the other tasks. Pool+CV calibrates mappings before averaging; PoolDir calibrates the pooled vector.")
    print("Oracle fits the held-out target itself and is diagnostic only.")


def run_self_test():
    for task in [normalized_task(task) for task in TASKS]:
        mappings = create_balanced_mappings(task["outcomes"], 3, 1234)
        verify_balanced_mappings(mappings, task["outcomes"], 3)
    random = mulberry32(1234)
    expected = [0.07329497812315822, 0.7034119898453355, 0.9028560190927237]
    actual = [random() for _ in expected]
    if any(abs(left - right) > 1e-15 for left, right in zip(actual, expected)):
        raise AssertionError(f"Mulberry32 parity failed: {actual}")
    target = [0.5, 0.3, 0.2]
    scale, oracle_tv = fit_oracle_scale([0.5, 0.3, 0.2], target)
    if abs(scale - 1.0) > 1e-12 or oracle_tv > 1e-12:
        raise AssertionError("Oracle scale identity failed")
    task = normalized_task(TASKS[2])
    fraction_response = """```json
{"maximum-1": 1/36, "maximum-2": 3/36, "maximum-3": 5/36,
 "maximum-4": 7/36, "maximum-5": 9/36, "maximum-6": 11/36}
```"""
    parsed = parse_probability_object(fraction_response, task)
    if total_variation(parsed, target_probabilities(task)) > 1e-12:
        raise AssertionError("Fractional probability response parsing failed")
    marker_response = "\n".join(
        f"PROB {outcome['id']} = {outcome['weight']}/24" for outcome in task["outcomes"]
    )
    marker_target = normalize([outcome["weight"] for outcome in task["outcomes"]])
    if total_variation(parse_probability_object(marker_response, task), marker_target) > 1e-12:
        raise AssertionError("Probability marker parsing failed")
    print("Self-test passed")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "infer", "procedural", "analyze", "self-test"])
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Running Logit Scope API base URL")
    parser.add_argument("--input", type=pathlib.Path, help="Collected dataset JSON for analyze mode")
    parser.add_argument("--output", type=pathlib.Path, help="Dataset JSON destination for run mode")
    parser.add_argument("--report-output", type=pathlib.Path, help="Optional analysis JSON destination")
    parser.add_argument("--tasks", default="all", help="Comma-separated task IDs, or all")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--assistant-prefix", default="Answer: ")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--procedural-count", type=int, default=20)
    parser.add_argument("--inference-thinking", action="store_true", help="Allow Qwen-style reasoning during probability inference")
    args = parser.parse_args()
    if args.mode in {"run", "procedural"} and args.output is None:
        parser.error(f"--output is required in {args.mode} mode")
    if args.mode in {"infer", "analyze"} and args.input is None:
        parser.error(f"--input is required in {args.mode} mode")
    if args.mode == "infer" and args.output is None:
        parser.error(f"--output is required in {args.mode} mode")
    if args.blocks < 1 or args.blocks > 10:
        parser.error("--blocks must be between 1 and 10")
    if args.procedural_count < 1 or args.procedural_count > 100:
        parser.error("--procedural-count must be between 1 and 100")
    return args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_arguments()
    if args.mode == "self-test":
        run_self_test()
        return
    if args.mode == "run":
        dataset = collect_dataset(args)
        write_json(args.output, dataset)
        print(f"Wrote {args.output}")
    elif args.mode == "procedural":
        dataset = add_probability_inferences(create_procedural_dataset(args), args)
        write_json(args.output, dataset)
        print(f"Wrote {args.output}")
    else:
        dataset = json.loads(args.input.read_text(encoding="utf-8"))
        if args.mode == "infer":
            dataset = add_probability_inferences(dataset, args)
            write_json(args.output, dataset)
            print(f"Wrote {args.output}")
    if dataset.get("probes"):
        report = analyze_dataset(dataset)
        print_report(report)
    else:
        report = analyze_inference_suite(dataset)
        print_inference_suite_report(report)
    if args.report_output:
        write_json(args.report_output, report)
        print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
