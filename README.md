# Logit Scope

Logit Scope is a local chat laboratory for reshaping an LLM's next-token probability distribution in real time. It preserves the model's candidate ranking, then applies a chosen rank profile while you watch the raw and shaped curves.

Created by [Aaron (`caustik`)](https://github.com/caustik) and released by APU Software, LLC.

![Logit Scope reshaping ranked next-token probabilities at 50% diversity](docs/images/logit-scope.png)

The project is a small C++ application built directly on [llama.cpp](https://github.com/ggml-org/llama.cpp). It serves an embedded, dependency-free web interface on localhost. There is no JUCE dependency, package manager, cloud service, or separate frontend build.

## What you can manipulate

- **Profile:** no shaping, or an exponential, power, or half-normal response by candidate rank
- **Diversity:** moves from deterministic sampling at 0%, through unchanged sampling at 100%, to the uniform limit over the selected pool at 200%, subject to the protocol guard
- **Pool:** truncates sampling to the top 32–4096 candidates
- **Seed:** initializes the random sampler for the next response
- **Protocol guard:** leaves control and end-of-generation logits at their raw values

Profile, diversity, pool, and protocol guard changes take effect on the next sampled token, including during a response. Seed changes take effect when the next response starts.

![Sweeping the Power profile from deterministic through raw to uniform sampling](docs/images/logit-scope-demo.gif)

## Clone and build

Prerequisites are Git, CMake 3.21 or newer, and a C++17 compiler. Ninja is used by the supplied macOS/Linux presets. On Windows, the preset uses Visual Studio 2022.

```sh
git clone --recurse-submodules https://github.com/caustik/logit-scope.git
cd logit-scope
```

If you already cloned without submodules:

```sh
git submodule update --init --recursive
```

Windows CPU build:

```powershell
cmake --preset windows-cpu-release
cmake --build --preset windows-cpu-release
ctest --preset windows-cpu-release
```

macOS or Linux CPU build:

```sh
cmake --preset unix-cpu-release
cmake --build --preset unix-cpu-release
ctest --preset unix-cpu-release
```

The Windows executable is `build/windows-cpu-release/Release/logit-scope.exe`. The macOS/Linux executable is `build/unix-cpu-release/logit-scope`.

Vulkan and CUDA presets are also provided; list the presets available on your platform with `cmake --list-presets`. A normal llama.cpp GPU toolchain is required for those configurations. macOS uses llama.cpp's Metal backend when available.

## Run

Supply any chat-capable GGUF model whose license permits your use:

```sh
logit-scope --model /path/to/model.gguf
```

The program opens `http://127.0.0.1:8080/` in your browser. It runs entirely on your machine and binds only to localhost by default.

Useful options:

```text
--ctx-size <tokens>   Context capacity (default: 4096)
--max-tokens <count>  Maximum tokens in one response (default: 1024)
--threads <count>     CPU inference threads
--gpu-layers <count>  Layers to offload (use a large value such as 999 for all)
--port <number>       Local HTTP port (default: 8080)
--no-browser          Do not open the browser automatically
```

You can also set `LOGIT_SCOPE_MODEL` instead of passing `--model` each time.

## How shaping works

At each sampling step, Logit Scope sorts the top `K` finite logits from highest to lowest. A shaped profile supplies a monotonically increasing rank curve `f(r)`:

```text
exponential:  f(r) = r
power:        f(r) = log(r + 1)
half-normal:  f(r) = r²

sharpen: l'(r) = l(r) - s f(r)
loosen:  l'(r) = l(r) + s f(r), clamped so l'(r) <= l'(r - 1)
```

The profile strength `s` is not exposed as a control. Instead, the shared calibrator solves for it on every token. Below 100%, diversity scales the raw entropy toward zero. Above 100%, it covers the remaining distance from raw entropy to the maximum entropy of a uniform distribution over the pool:

```text
0% ≤ diversity ≤ 100%: shaped entropy = D × raw entropy
100% < diversity ≤ 200%: shaped entropy = raw entropy + (D - 1) × (log(K) - raw entropy)
effective choices = exp(entropy)
```

Below 100%, the profile removes probability from lower ranks. Above 100%, it returns probability toward lower ranks while preserving the model's candidate order; 150% is halfway from the raw entropy to the pool's maximum entropy, and 200% is uniform before any protocol-guard restoration. This keeps profile experiments small: a new profile supplies one rank curve, while the common bidirectional calibration and UI contract remain unchanged. The **None** profile is an exact shaping bypass, and 100% diversity is also an exact shaping bypass, although the selected pool still acts as top-K truncation.

Zero diversity is handled as an explicit boundary rather than asking the numerical calibrator to find infinite strength. The rank shaper assigns zero probability to every candidate below rank one. With **Protocol guard** enabled, protected control and end-of-generation logits are then restored and the surrounding logits are clamped to preserve candidate order, so the result is as deterministic as the guard permits rather than necessarily having exactly zero entropy.

Before generation, the scope shows a clearly labeled illustrative rank curve and runs it through the same C++ shaper as real tokens, so profile, diversity, and pool changes have immediate visual feedback. The preview is not a model prediction and does not include token-specific protocol-guard effects. During generation, actual token probabilities replace the preview and the scope reports pool mass, Jensen–Shannon shift, effective choices, peak probability, and the selected token. Sampling data is published only after a non-EOG token is selected. While generation is active the plot follows the current decision; after completion it retains the response's highest-uncertainty decision instead of ending on almost-certain punctuation or EOG. The probability axis automatically follows the meaningful six-decade range for the displayed token. Its `log(1 + r)` horizontal axis spans the full pool: the left endpoint is the top-ranked candidate, and the right endpoint is the final candidate, labeled with the pool size `K`. Up to 64 display samples are distributed across that domain rather than truncating the view to the first 64 candidates.

## Project layout

```text
include/logit_scope/   Rank shaping and engine API
src/                   llama.cpp sampler, chat engine, and local HTTP server
web/                   Embedded HTML, CSS, and JavaScript UI
tests/                 Dependency-free rank-profile tests
third_party/llama.cpp  Pinned llama.cpp Git submodule
```

The frontend files are converted to C++ byte arrays during the CMake build and compiled into the executable.

## Current MVP boundaries

- One local user/session and one model per process
- Text-only GGUF chat models
- Context is retained until it fills; use **Clear** to start over
- No model downloader or model redistribution
- No authentication when binding to a non-localhost address; doing so is not recommended on an untrusted network

## Licensing

The original Logit Scope source code is licensed under the [Apache License 2.0](LICENSE), with copyright held by APU Software, LLC. Creator attribution is recorded in [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff).

llama.cpp and the libraries vendored within its submodule retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). GGUF model weights are not included and may have separate terms.
