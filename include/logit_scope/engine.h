#pragma once

#include "logit_scope/rank_profile.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace logit_scope
{

constexpr std::size_t display_rank_count = 64;

struct EngineConfig
{
    std::string model_path;
    int context_size = 4096;
    int maximum_response_tokens = 1024;
    int threads = 0;
    int gpu_layers = 0;
};

struct SamplingSnapshot
{
    bool model_loaded = false;
    bool generating = false;
    std::string status = "Starting llama.cpp...";
    std::string transcript;
    std::string selected_token;
    int sampling_step = 0;
    bool representative_sampling = false;
    std::size_t candidate_count = 0;
    std::size_t probability_count = 0;
    std::array<std::size_t, display_rank_count> probability_ranks{};
    std::array<float, display_rank_count> raw_probabilities{};
    std::array<float, display_rank_count> shaped_probabilities{};
    float raw_entropy = 0.0f;
    float shaped_entropy = 0.0f;
    float raw_peak_probability = 0.0f;
    float shaped_peak_probability = 0.0f;
    float pool_probability_mass = 0.0f;
    float jensen_shannon_divergence = 0.0f;
};

class Engine
{
  public:
    explicit Engine(EngineConfig config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void start();
    void stop();
    bool submit_message(std::string message);
    void cancel_generation();
    void clear_conversation();

    ShapeSettings shape_settings() const;
    void set_shape_settings(const ShapeSettings& settings);
    SamplingSnapshot snapshot() const;
    SamplingSnapshot preview_snapshot(const ShapeSettings& settings) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace logit_scope
