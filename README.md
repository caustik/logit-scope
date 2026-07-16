# Logit Scope

Logit Scope is a local chat laboratory for reshaping an LLM's next-token probability distribution in real time. It preserves the model's candidate ranking, then blends the raw probabilities with a chosen rank profile while you watch the raw, target, and resulting curves.

Created by [Aaron (`caustik`)](https://github.com/caustik) and released by APU Software, LLC.

![Logit Scope reshaping ranked next-token probabilities during a local conversation](docs/images/logit-scope.png)

The project is a small C++ application built directly on [llama.cpp](https://github.com/ggml-org/llama.cpp). It serves an embedded, dependency-free web interface on localhost. There is no JUCE dependency, package manager, cloud service, or separate frontend build.

## What you can manipulate

- **Profile:** uniform, exponential, power, or half-normal decay by candidate rank
- **Blend:** interpolates from the model's raw distribution (`0%`) to the target profile (`100%`)
- **Concentration:** controls how quickly the target probability decays with rank
- **Pool:** truncates sampling to the top 32–4096 candidates
- **Seed:** initializes the random sampler for the next response
- **Protocol guard:** leaves control and end-of-generation logits at their raw values

Profile, blend, concentration, pool, and protocol guard changes take effect on the next sampled token, including during a response. Seed changes take effect when the next response starts.

![Changing the Power profile concentration while the conversation response is generated](docs/images/logit-scope-demo.gif)

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

At each sampling step, Logit Scope sorts the top `K` finite logits from highest to lowest. It evaluates a monotonically non-increasing target probability `q(r)` at each zero-based rank `r`:

```text
uniform:      q(r) ∝ 1
exponential:  q(r) ∝ exp(-c r)
power:        q(r) ∝ (r + 1)^(-c)
half-normal:  q(r) ∝ exp(-0.5 c² r²)
```

The raw rank probability `p(r)` and target are blended geometrically:

```text
p'(r) ∝ p(r)^(1-b) q(r)^b
```

Here `b` is Blend and `c` is Concentration. Since both inputs are non-increasing by rank, the transformed logits preserve candidate order. `Blend = 0` is an exact shaping bypass, although the selected pool still acts as top-K truncation.

The scope remains visible while idle so you can see the target shape before generating. During generation it overlays the raw and shaped distributions and reports pool mass, Jensen–Shannon shift, entropy, peak probability, and the selected token.

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
