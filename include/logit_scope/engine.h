#pragma once

#include "logit_scope/rank_profile.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <array>

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
    ShapeSettings sampling_settings;
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

struct LogitLandscapeCandidate
{
    int token_id = -1;
    std::string text;
    int rank = -1;
    float relative_logit = 0.0f;
    bool protected_token = false;
};

struct LogitLandscape
{
    bool available = false;
    int sampling_step = 0;
    int selected_token_id = -1;
    std::string selected_token;
    ShapeSettings sampling_settings;
    std::size_t finite_candidate_count = 0;
    double captured_probability_mass = 0.0;
    std::vector<LogitLandscapeCandidate> candidates;
};

struct EvaluationResult
{
    std::uint64_t id = 0;
    bool generating = false;
    bool ready = false;
    std::string response;
    std::string status;
    int token_count = 0;
};

struct DistributionOutcomeRequest
{
    std::string id;
    std::string text;
    std::string label;
    double target_probability = 0.0;
};

struct DistributionConfigurationRequest
{
    std::string id;
    std::string name;
    ShapeSettings settings;
};

struct DistributionProbeRequest
{
    std::string prompt;
    std::string assistant_prefix = "Answer: ";
    bool auto_select_assistant_prefix = true;
    std::string mapping_id;
    std::vector<DistributionOutcomeRequest> outcomes;
    std::vector<DistributionConfigurationRequest> configurations;
    std::size_t sample_count = 10000;
    std::uint32_t seed = 1234;
};

struct DistributionDistanceMetrics
{
    bool valid = true;
    double total_variation = 0.0;
    double js_divergence_nats = 0.0;
    double js_distance_normalized = 0.0;
    double valid_mass = 0.0;
    double invalid_mass = 0.0;
    double conditional_entropy = 0.0;
    double conditional_entropy_error = 0.0;
    double pairwise_order_accuracy = 0.0;
    double missing_target_mass = 0.0;
};

struct DistributionOutcomeResult
{
    std::string id;
    std::string text;
    std::string label;
    int token_id = -1;
    double target_probability = 0.0;
    double probability = 0.0;
    bool retained = false;
    int rank = -1;
    std::uint64_t projected_count = 0;
};

struct DistributionTokenResult
{
    int token_id = -1;
    std::string text;
    double probability = 0.0;
    int rank = -1;
};

struct DistributionStageResult
{
    std::vector<DistributionOutcomeResult> outcomes;
    std::vector<DistributionTokenResult> top_invalid_tokens;
    DistributionDistanceMetrics open_metrics;
    DistributionDistanceMetrics conditional_metrics;
    std::uint64_t projected_invalid_count = 0;
};

struct DistributionSamplerDiagnostics
{
    int support_size = 0;
    double retained_probability_mass = 0.0;
    double sampler_raw_entropy = 0.0;
    double sampler_target_entropy = 0.0;
    double sampler_shaped_entropy = 0.0;
    double sampler_entropy_error = 0.0;
    double raw_effective_choices = 0.0;
    double target_effective_choices = 0.0;
    double shaped_effective_choices = 0.0;
    bool target_saturated = false;
};

struct DistributionConfigurationResult
{
    std::string id;
    std::string name;
    ShapeSettings settings;
    DistributionStageResult retained_raw;
    DistributionStageResult shaped;
    DistributionSamplerDiagnostics diagnostics;
};

struct DistributionProbeResult
{
    std::uint64_t id = 0;
    bool generating = false;
    bool ready = false;
    std::string status;
    std::string prompt;
    std::string formatted_prompt;
    std::string requested_assistant_prefix;
    std::string assistant_prefix;
    std::string label_token_prefix;
    bool assistant_prefix_auto_selected = false;
    std::string mapping_id;
    std::string model_path;
    int prompt_token_count = 0;
    std::size_t sample_count = 0;
    std::uint32_t seed = 0;
    double task_target_entropy = 0.0;
    DistributionStageResult full_raw;
    std::vector<DistributionConfigurationResult> configurations;
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
    std::uint64_t submit_evaluation(std::string prompt, const ShapeSettings& settings, std::string assistant_prefix = {});
    std::uint64_t submit_distribution(DistributionProbeRequest request);
    void cancel_generation();
    void clear_conversation();

    ShapeSettings shape_settings() const;
    void set_shape_settings(const ShapeSettings& settings);
    SamplingSnapshot snapshot() const;
    SamplingSnapshot preview_snapshot(const ShapeSettings& settings) const;
    LogitLandscape logit_landscape() const;
    EvaluationResult evaluation_result() const;
    DistributionProbeResult distribution_result() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace logit_scope
