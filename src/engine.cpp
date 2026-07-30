#include "logit_scope/engine.h"
#include "logit_scope/distribution_metrics.h"

#include <llama.h>

#include <condition_variable>
#include <unordered_set>
#include <string_view>
#include <filesystem>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <utility>
#include <atomic>
#include <cctype>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#include <cmath>
#include <deque>
#include <mutex>

namespace logit_scope
{
namespace
{

struct ChatMessage
{
    std::string role;
    std::string content;
};

std::string sanitize_utf8(std::string_view bytes)
{
    constexpr std::string_view replacement = "\xef\xbf\xbd";
    std::string result;
    result.reserve(bytes.size());

    std::size_t index = 0;
    while (index < bytes.size())
    {
        const auto first = static_cast<unsigned char>(bytes[index]);
        if (first < 0x80)
        {
            result.push_back(static_cast<char>(first));
            ++index;
            continue;
        }

        std::size_t length = 0;
        std::uint32_t code_point = 0;
        std::uint32_t minimum = 0;
        if ((first & 0xe0) == 0xc0)
        {
            length = 2;
            code_point = first & 0x1f;
            minimum = 0x80;
        }
        else if ((first & 0xf0) == 0xe0)
        {
            length = 3;
            code_point = first & 0x0f;
            minimum = 0x800;
        }
        else if ((first & 0xf8) == 0xf0)
        {
            length = 4;
            code_point = first & 0x07;
            minimum = 0x10000;
        }

        bool valid = length != 0 && index + length <= bytes.size();
        for (std::size_t offset = 1; valid && offset < length; ++offset)
        {
            const auto continuation = static_cast<unsigned char>(bytes[index + offset]);
            valid = (continuation & 0xc0) == 0x80;
            if (valid) code_point = (code_point << 6) | (continuation & 0x3f);
        }

        valid = valid && code_point >= minimum && code_point <= 0x10ffff && !(code_point >= 0xd800 && code_point <= 0xdfff);
        if (!valid)
        {
            result.append(replacement);
            ++index;
            continue;
        }

        result.append(bytes.substr(index, length));
        index += length;
    }

    return result;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token)
{
    char buffer[256]{};
    const auto length = llama_token_to_piece(vocab, token, buffer, static_cast<int32_t>(sizeof(buffer)), 0, true);
    if (length >= 0) return std::string(buffer, static_cast<std::size_t>(length));

    std::string piece(static_cast<std::size_t>(-length), '\0');
    const auto actual_length = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (actual_length < 0) return {};
    piece.resize(static_cast<std::size_t>(actual_length));
    return piece;
}

} // namespace

class Engine::Impl
{
  public:
    explicit Impl(EngineConfig config) : config_(std::move(config)) {}

    ~Impl() { stop(); }

    void start()
    {
        if (started_.exchange(true)) return;
        worker_ = std::thread([this] { run(); });
    }

    void stop()
    {
        if (!started_.exchange(false)) return;
        stop_requested_.store(true);
        queue_condition_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    bool submit_message(std::string message)
    {
        message.erase(message.begin(),
                      std::find_if(message.begin(), message.end(), [](unsigned char value) { return !std::isspace(value); }));
        while (!message.empty() && std::isspace(static_cast<unsigned char>(message.back()))) message.pop_back();
        if (message.empty()) return false;

        {
            const std::lock_guard offline_job_lock(offline_job_mutex_);
            const std::lock_guard snapshot_lock(snapshot_mutex_);
            if (offline_job_type_ != OfflineJobType::none || !snapshot_.model_loaded || snapshot_.generating) return false;
            snapshot_.generating = true;
            snapshot_.status = "Queued...";
        }

        {
            const std::lock_guard queue_lock(queue_mutex_);
            pending_messages_.push_back(std::move(message));
        }
        cancel_requested_.store(false);
        queue_condition_.notify_one();
        return true;
    }

    std::uint64_t submit_evaluation(std::string prompt, const ShapeSettings& settings, std::string assistant_prefix)
    {
        prompt.erase(prompt.begin(), std::find_if(prompt.begin(), prompt.end(), [](unsigned char value) { return !std::isspace(value); }));
        while (!prompt.empty() && std::isspace(static_cast<unsigned char>(prompt.back()))) prompt.pop_back();
        if (prompt.empty() || assistant_prefix.size() > 512) return 0;

        EvaluationRequest request;
        request.id = next_evaluation_id_.fetch_add(1);
        request.prompt = std::move(prompt);
        request.assistant_prefix = std::move(assistant_prefix);
        request.settings = settings;
        request.restoration_settings = shape_settings();
        {
            const std::lock_guard offline_job_lock(offline_job_mutex_);
            if (offline_job_type_ != OfflineJobType::none) return 0;
            offline_job_type_ = OfflineJobType::blind_evaluation;
            const std::lock_guard snapshot_lock(snapshot_mutex_);
            if (!snapshot_.model_loaded || snapshot_.generating)
            {
                offline_job_type_ = OfflineJobType::none;
                return 0;
            }
            request.restoration_snapshot = snapshot_;
            snapshot_.generating = true;
            snapshot_.status = "Evaluating...";
        }
        {
            const std::lock_guard evaluation_lock(evaluation_mutex_);
            evaluation_result_ = {};
            evaluation_result_.id = request.id;
            evaluation_result_.generating = true;
            evaluation_result_.status = "Generating blinded response...";
        }
        const auto id = request.id;
        {
            const std::lock_guard queue_lock(queue_mutex_);
            pending_evaluations_.push_back(std::move(request));
        }
        cancel_requested_.store(false);
        queue_condition_.notify_one();
        return id;
    }

    std::uint64_t submit_distribution(DistributionProbeRequest request)
    {
        validate_distribution_request(request);

        PendingDistributionProbe pending;
        pending.id = next_distribution_id_.fetch_add(1);
        pending.request = std::move(request);
        {
            const std::lock_guard offline_job_lock(offline_job_mutex_);
            if (offline_job_type_ != OfflineJobType::none) return 0;
            offline_job_type_ = OfflineJobType::distribution_probe;
            const std::lock_guard snapshot_lock(snapshot_mutex_);
            if (!snapshot_.model_loaded || snapshot_.generating)
            {
                offline_job_type_ = OfflineJobType::none;
                return 0;
            }
            pending.restoration_snapshot = snapshot_;
            snapshot_.generating = true;
            snapshot_.status = "Running Distribution Lab probe...";
        }
        {
            const std::lock_guard distribution_lock(distribution_mutex_);
            distribution_result_ = {};
            distribution_result_.id = pending.id;
            distribution_result_.generating = true;
            distribution_result_.status = "Preparing distribution probe...";
            distribution_result_.mapping_id = pending.request.mapping_id;
        }
        const auto id = pending.id;
        {
            const std::lock_guard queue_lock(queue_mutex_);
            pending_distributions_.push_back(std::move(pending));
        }
        cancel_requested_.store(false);
        queue_condition_.notify_one();
        return id;
    }

    void cancel_generation()
    {
        cancel_requested_.store(true);
        update_snapshot(
            [](SamplingSnapshot& snapshot)
            {
                if (snapshot.generating) snapshot.status = "Stopping...";
            });
    }

    void clear_conversation()
    {
        reset_requested_.store(true);
        cancel_requested_.store(true);
        queue_condition_.notify_all();
    }

    ShapeSettings shape_settings() const
    {
        ShapeSettings settings;
        settings.profile = static_cast<RankProfile>(settings_profile_.load());
        settings.diversity = settings_diversity_.load();
        settings.candidate_cap = settings_candidate_cap_.load();
        settings.minimum_relative_probability = settings_minimum_relative_probability_.load();
        settings.seed = settings_seed_.load();
        settings.protect_control_tokens = settings_protect_control_.load();
        return settings;
    }

    void set_shape_settings(const ShapeSettings& settings)
    {
        const auto candidate_cap = std::max<std::size_t>(2, std::min(maximum_candidate_cap, settings.candidate_cap));
        settings_profile_.store(static_cast<int>(settings.profile));
        settings_diversity_.store(std::max(0.0f, std::min(maximum_diversity, settings.diversity)));
        settings_candidate_cap_.store(candidate_cap);
        settings_minimum_relative_probability_.store(std::max(0.0f, std::min(1.0f, settings.minimum_relative_probability)));
        settings_seed_.store(settings.seed);
        settings_protect_control_.store(settings.protect_control_tokens);
    }

    SamplingSnapshot snapshot() const
    {
        const std::lock_guard lock(snapshot_mutex_);
        return snapshot_;
    }

    SamplingSnapshot preview_snapshot(const ShapeSettings& settings) const
    {
        const std::lock_guard lock(preview_mutex_);
        if (preview_valid_ && settings.profile == preview_settings_.profile && settings.diversity == preview_settings_.diversity &&
            settings.candidate_cap == preview_settings_.candidate_cap &&
            settings.minimum_relative_probability == preview_settings_.minimum_relative_probability)
            return preview_snapshot_;

        SamplingSnapshot preview;
        std::vector<float> raw_logits(std::max<std::size_t>(2, std::min(maximum_candidate_cap, settings.candidate_cap)));
        for (std::size_t rank = 0; rank < raw_logits.size(); ++rank)
            raw_logits[rank] = static_cast<float>(-1.2 * std::log(static_cast<double>(rank) + 1.0));
        apply_relative_probability_floor(raw_logits, settings.minimum_relative_probability);
        preview.candidate_count = raw_logits.size();
        auto shaped_logits = raw_logits;
        shape_ranked_logits(shaped_logits, settings);

        ProbabilityMetrics raw_metrics;
        ProbabilityMetrics shaped_metrics;
        const auto raw_probabilities = probabilities_from_logits(raw_logits, &raw_metrics);
        const auto shaped_probabilities = probabilities_from_logits(shaped_logits, &shaped_metrics);
        preview.probability_count = std::min(display_rank_count, preview.candidate_count);
        auto previous_rank = std::size_t{};
        for (std::size_t display_index = 0; display_index < preview.probability_count; ++display_index)
        {
            const auto rank = display_rank_at(display_index, preview.probability_count, preview.candidate_count, previous_rank);
            preview.probability_ranks[display_index] = rank;
            preview.raw_probabilities[display_index] = raw_probabilities[rank];
            preview.shaped_probabilities[display_index] = shaped_probabilities[rank];
            previous_rank = rank;
        }
        preview.raw_entropy = raw_metrics.entropy;
        preview.shaped_entropy = shaped_metrics.entropy;
        preview.raw_peak_probability = raw_metrics.peak_probability;
        preview.shaped_peak_probability = shaped_metrics.peak_probability;
        preview.pool_probability_mass = 1.0f;
        preview.jensen_shannon_divergence = jensen_shannon_divergence(raw_probabilities, shaped_probabilities);
        preview_snapshot_ = preview;
        preview_settings_ = settings;
        preview_valid_ = true;
        return preview_snapshot_;
    }

    EvaluationResult evaluation_result() const
    {
        const std::lock_guard lock(evaluation_mutex_);
        return evaluation_result_;
    }

    DistributionProbeResult distribution_result() const
    {
        const std::lock_guard lock(distribution_mutex_);
        return distribution_result_;
    }

  private:
    enum class OfflineJobType
    {
        none,
        blind_evaluation,
        distribution_probe,
    };

    struct SamplerContext
    {
        Impl* engine = nullptr;
    };

    struct PendingSamplingData
    {
        bool valid = false;
        std::string selected_token;
        ShapeSettings settings;
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

    struct EvaluationRequest
    {
        std::uint64_t id = 0;
        std::string prompt;
        std::string assistant_prefix;
        ShapeSettings settings;
        ShapeSettings restoration_settings;
        SamplingSnapshot restoration_snapshot;
    };

    struct PendingDistributionProbe
    {
        std::uint64_t id = 0;
        DistributionProbeRequest request;
        SamplingSnapshot restoration_snapshot;
    };

    struct CandidateTransformDiagnostics
    {
        std::vector<int> retained_token_ids;
        std::vector<double> retained_raw_probabilities;
        std::vector<double> shaped_probabilities;
        std::vector<int> retained_ranks;
        double retained_probability_mass = 0.0;
        ProbabilityMetrics raw_metrics;
        ProbabilityMetrics target_metrics;
        ProbabilityMetrics shaped_metrics;
        bool target_saturated = false;
    };

    void run()
    {
        if (!initialize_model()) return;

        while (!stop_requested_.load())
        {
            std::string message;
            EvaluationRequest evaluation;
            PendingDistributionProbe distribution;
            {
                std::unique_lock lock(queue_mutex_);
                queue_condition_.wait(lock,
                                      [this]
                                      {
                                          return stop_requested_.load() || reset_requested_.load() || !pending_messages_.empty() ||
                                                 !pending_evaluations_.empty() || !pending_distributions_.empty();
                                      });
                if (stop_requested_.load()) break;

                if (reset_requested_.exchange(false))
                {
                    pending_messages_.clear();
                    pending_evaluations_.clear();
                    pending_distributions_.clear();
                    {
                        const std::lock_guard evaluation_lock(evaluation_mutex_);
                        evaluation_result_ = {};
                    }
                    {
                        const std::lock_guard distribution_lock(distribution_mutex_);
                        distribution_result_ = {};
                    }
                    {
                        const std::lock_guard offline_job_lock(offline_job_mutex_);
                        offline_job_type_ = OfflineJobType::none;
                    }
                    pending_sampling_ = {};
                    representative_sampling_ = {};
                    messages_.clear();
                    current_response_.clear();
                    previous_formatted_length_ = 0;
                    cancel_requested_.store(false);
                    if (context_ != nullptr) llama_memory_clear(llama_get_memory(context_), true);
                    update_snapshot(
                        [](SamplingSnapshot& snapshot)
                        {
                            const auto loaded = snapshot.model_loaded;
                            snapshot = {};
                            snapshot.model_loaded = loaded;
                            snapshot.status = loaded ? "Ready - conversation cleared" : "Model is not loaded";
                        });
                }

                if (!pending_messages_.empty())
                {
                    message = std::move(pending_messages_.front());
                    pending_messages_.pop_front();
                }
                else if (!pending_evaluations_.empty())
                {
                    evaluation = std::move(pending_evaluations_.front());
                    pending_evaluations_.pop_front();
                }
                else if (!pending_distributions_.empty())
                {
                    distribution = std::move(pending_distributions_.front());
                    pending_distributions_.pop_front();
                }
            }

            if (!message.empty())
                process_message(message);
            else if (evaluation.id != 0)
                process_evaluation(std::move(evaluation));
            else if (distribution.id != 0)
                process_distribution_probe(std::move(distribution));
        }

        release_model();
    }

    void process_evaluation(EvaluationRequest request)
    {
        auto saved_messages = std::move(messages_);
        auto saved_response = std::move(current_response_);
        const auto saved_pending_sampling = pending_sampling_;
        const auto saved_representative_sampling = representative_sampling_;

        messages_.clear();
        current_response_.clear();
        pending_sampling_ = {};
        representative_sampling_ = {};
        previous_formatted_length_ = 0;
        if (context_ != nullptr) llama_memory_clear(llama_get_memory(context_), true);
        set_shape_settings(request.settings);

        process_message(request.prompt, request.assistant_prefix);
        const auto generated_snapshot = snapshot();

        EvaluationResult result;
        result.id = request.id;
        result.ready = true;
        result.response = sanitize_utf8(current_response_);
        result.status = generated_snapshot.status;
        result.token_count = generated_snapshot.sampling_step;

        messages_ = std::move(saved_messages);
        current_response_ = std::move(saved_response);
        pending_sampling_ = saved_pending_sampling;
        representative_sampling_ = saved_representative_sampling;
        set_shape_settings(request.restoration_settings);
        previous_formatted_length_ = 0;
        if (context_ != nullptr) llama_memory_clear(llama_get_memory(context_), true);
        {
            const std::lock_guard snapshot_lock(snapshot_mutex_);
            snapshot_ = std::move(request.restoration_snapshot);
        }
        {
            const std::lock_guard evaluation_lock(evaluation_mutex_);
            evaluation_result_ = std::move(result);
        }
        {
            const std::lock_guard offline_job_lock(offline_job_mutex_);
            offline_job_type_ = OfflineJobType::none;
        }
    }

    static void validate_distribution_request(const DistributionProbeRequest& request)
    {
        const auto has_non_whitespace = [](const std::string& text)
        { return std::any_of(text.begin(), text.end(), [](unsigned char value) { return !std::isspace(value); }); };
        if (!has_non_whitespace(request.prompt)) throw std::invalid_argument("Distribution prompt is empty");
        if (request.assistant_prefix.empty() || request.assistant_prefix.size() > 128)
            throw std::invalid_argument("Assistant prefix must contain 1 through 128 bytes");
        if (request.mapping_id.empty()) throw std::invalid_argument("Mapping ID is empty");
        if (request.outcomes.size() < 2 || request.outcomes.size() > 26)
            throw std::invalid_argument("Distribution probes require 2 through 26 outcomes");
        if (request.configurations.empty() || request.configurations.size() > 4)
            throw std::invalid_argument("Distribution probes require 1 through 4 configurations");
        if (request.sample_count > 1000000) throw std::invalid_argument("Projected sample count exceeds 1,000,000");

        std::unordered_set<std::string> outcome_ids;
        std::unordered_set<std::string> labels;
        double target_total = 0.0;
        for (const auto& outcome : request.outcomes)
        {
            if (outcome.id.empty() || outcome.label.empty()) throw std::invalid_argument("Outcome IDs and labels must not be empty");
            if (!outcome_ids.insert(outcome.id).second) throw std::invalid_argument("Outcome IDs must be unique");
            if (!labels.insert(outcome.label).second) throw std::invalid_argument("Outcome labels must be unique");
            if (!std::isfinite(outcome.target_probability) || outcome.target_probability < 0.0 || outcome.target_probability > 1.0)
                throw std::invalid_argument("Target probabilities must be finite values from 0 through 1");
            target_total += outcome.target_probability;
        }
        if (std::abs(target_total - 1.0) > 1.0e-6) throw std::invalid_argument("Target probabilities must sum to 1");

        std::unordered_set<std::string> configuration_ids;
        for (const auto& configuration : request.configurations)
        {
            if (configuration.id.empty() || configuration.name.empty())
                throw std::invalid_argument("Configuration IDs and names must not be empty");
            if (!configuration_ids.insert(configuration.id).second) throw std::invalid_argument("Configuration IDs must be unique");
            const auto& settings = configuration.settings;
            if (!std::isfinite(settings.diversity) || settings.diversity < 0.0f || settings.diversity > maximum_diversity)
                throw std::invalid_argument("Configuration diversity is outside the supported range");
            if (settings.candidate_cap < 2 || settings.candidate_cap > maximum_candidate_cap)
                throw std::invalid_argument("Configuration candidate cap is outside the supported range");
            if (!std::isfinite(settings.minimum_relative_probability) || settings.minimum_relative_probability < 0.0f ||
                settings.minimum_relative_probability > 1.0f)
                throw std::invalid_argument("Configuration Min-P floor is outside the supported range");
        }
    }

    static void copy_metric_result(const DistributionMetricResult& source, DistributionDistanceMetrics& destination)
    {
        destination.total_variation = source.total_variation;
        destination.js_divergence_nats = source.js_divergence_nats;
        destination.js_distance_normalized = source.js_distance_normalized;
        destination.conditional_entropy = source.entropy;
        destination.conditional_entropy_error = source.entropy_error;
        destination.pairwise_order_accuracy = source.pairwise_order_accuracy;
        destination.missing_target_mass = source.missing_target_mass;
    }

    static void calculate_stage_metrics(DistributionStageResult& stage, double target_entropy)
    {
        std::vector<double> predicted;
        std::vector<double> target;
        std::vector<bool> retained;
        predicted.reserve(stage.outcomes.size());
        target.reserve(stage.outcomes.size());
        retained.reserve(stage.outcomes.size());
        for (const auto& outcome : stage.outcomes)
        {
            predicted.push_back(outcome.probability);
            target.push_back(outcome.target_probability);
            retained.push_back(outcome.retained);
        }

        const auto valid_mass = std::accumulate(predicted.begin(), predicted.end(), 0.0);
        const auto invalid_mass = std::max(0.0, 1.0 - valid_mass);
        auto predicted_open = predicted;
        auto target_open = target;
        auto retained_open = retained;
        predicted_open.push_back(invalid_mass);
        target_open.push_back(0.0);
        retained_open.push_back(true);
        copy_metric_result(DistributionMetrics::calculate(predicted_open, target_open, retained_open, target_entropy), stage.open_metrics);
        stage.open_metrics.valid_mass = valid_mass;
        stage.open_metrics.invalid_mass = invalid_mass;

        if (valid_mass <= std::numeric_limits<double>::min())
        {
            stage.conditional_metrics.valid = false;
            stage.conditional_metrics.valid_mass = 0.0;
            stage.conditional_metrics.invalid_mass = 1.0;
            stage.conditional_metrics.missing_target_mass = stage.open_metrics.missing_target_mass;
            stage.open_metrics.conditional_entropy = 0.0;
            stage.open_metrics.conditional_entropy_error = -target_entropy;
            stage.open_metrics.pairwise_order_accuracy = 0.0;
            return;
        }

        const auto conditional = DistributionMetrics::conditional_distribution(predicted);
        copy_metric_result(DistributionMetrics::calculate(conditional, target, retained, target_entropy), stage.conditional_metrics);
        stage.conditional_metrics.valid_mass = 1.0;
        stage.conditional_metrics.invalid_mass = 0.0;
        stage.open_metrics.conditional_entropy = stage.conditional_metrics.conditional_entropy;
        stage.open_metrics.conditional_entropy_error = stage.conditional_metrics.conditional_entropy_error;
        stage.open_metrics.pairwise_order_accuracy = stage.conditional_metrics.pairwise_order_accuracy;
    }

    std::string token_display_text(int token_id) const
    {
        auto text = sanitize_utf8(token_to_piece(vocab_, token_id));
        std::string display;
        display.reserve(text.size());
        for (const auto character : text)
        {
            switch (character)
            {
            case '\n':
                display += "\\n";
                break;
            case '\r':
                display += "\\r";
                break;
            case '\t':
                display += "\\t";
                break;
            default:
                display.push_back(character);
                break;
            }
        }
        return display.empty() ? "<empty>" : display;
    }

    DistributionStageResult create_distribution_stage(const DistributionProbeRequest& request, const std::vector<int>& label_token_ids,
                                                      const std::vector<int>& token_ids, const std::vector<double>& probabilities,
                                                      const std::vector<int>& ranks) const
    {
        DistributionStageResult stage;
        stage.outcomes.reserve(request.outcomes.size());
        for (std::size_t outcome_index = 0; outcome_index < request.outcomes.size(); ++outcome_index)
        {
            const auto& requested = request.outcomes[outcome_index];
            DistributionOutcomeResult outcome;
            outcome.id = requested.id;
            outcome.text = requested.text;
            outcome.label = requested.label;
            outcome.token_id = label_token_ids[outcome_index];
            outcome.target_probability = requested.target_probability;
            const auto found = std::find(token_ids.begin(), token_ids.end(), outcome.token_id);
            if (found != token_ids.end())
            {
                const auto index = static_cast<std::size_t>(std::distance(token_ids.begin(), found));
                outcome.probability = probabilities[index];
                outcome.retained = true;
                outcome.rank = ranks[index];
            }
            stage.outcomes.push_back(std::move(outcome));
        }

        const std::unordered_set<int> valid_tokens(label_token_ids.begin(), label_token_ids.end());
        for (std::size_t index = 0; index < token_ids.size() && stage.top_invalid_tokens.size() < 10; ++index)
        {
            if (valid_tokens.find(token_ids[index]) != valid_tokens.end()) continue;
            stage.top_invalid_tokens.push_back(
                {token_ids[index], token_display_text(token_ids[index]), probabilities[index], ranks[index]});
        }
        calculate_stage_metrics(stage, DistributionMetrics::entropy(
                                           [&request]
                                           {
                                               std::vector<double> target;
                                               target.reserve(request.outcomes.size());
                                               for (const auto& outcome : request.outcomes) target.push_back(outcome.target_probability);
                                               return target;
                                           }()));
        return stage;
    }

    static std::uint64_t projected_seed(std::uint32_t seed, const std::string& mapping_id, const std::string& configuration_id)
    {
        auto hash = std::uint64_t{14695981039346656037ull};
        const auto append = [&hash](const std::string& text)
        {
            for (const auto character : text)
            {
                hash ^= static_cast<unsigned char>(character);
                hash *= 1099511628211ull;
            }
            hash ^= 0xffu;
            hash *= 1099511628211ull;
        };
        append(mapping_id);
        append(configuration_id);
        return hash ^ static_cast<std::uint64_t>(seed);
    }

    static double next_unit_interval(std::mt19937_64& random) { return static_cast<double>(random() >> 11) * 0x1.0p-53; }

    static void create_projected_counts(DistributionStageResult& stage, std::size_t sample_count, std::uint64_t seed)
    {
        std::vector<double> cumulative;
        cumulative.reserve(stage.outcomes.size() + 1);
        double total = 0.0;
        for (const auto& outcome : stage.outcomes)
        {
            total += outcome.probability;
            cumulative.push_back(total);
        }
        total += std::max(0.0, 1.0 - total);
        cumulative.push_back(total);
        if (total <= std::numeric_limits<double>::min()) return;
        for (auto& value : cumulative) value /= total;
        cumulative.back() = 1.0;

        std::mt19937_64 random(seed);
        for (std::size_t sample = 0; sample < sample_count; ++sample)
        {
            const auto selected = std::upper_bound(cumulative.begin(), cumulative.end(), next_unit_interval(random));
            const auto index = static_cast<std::size_t>(std::distance(cumulative.begin(), selected));
            if (index < stage.outcomes.size())
                ++stage.outcomes[index].projected_count;
            else
                ++stage.projected_invalid_count;
        }
    }

    struct DistributionBoundaryCandidate
    {
        std::string assistant_prefix;
        std::string label_token_prefix;
    };

    static std::vector<DistributionBoundaryCandidate> distribution_boundary_candidates(const DistributionProbeRequest& request)
    {
        std::vector<DistributionBoundaryCandidate> candidates;
        const auto append_candidate = [&candidates](std::string assistant_prefix, std::string label_token_prefix = {})
        {
            if (assistant_prefix.empty() || assistant_prefix.size() > 128 ||
                std::find_if(
                    candidates.begin(), candidates.end(), [&assistant_prefix, &label_token_prefix](const auto& candidate)
                    { return candidate.assistant_prefix == assistant_prefix && candidate.label_token_prefix == label_token_prefix; }) !=
                    candidates.end())
                return;
            candidates.push_back({std::move(assistant_prefix), std::move(label_token_prefix)});
        };
        if (!request.auto_select_assistant_prefix)
        {
            append_candidate(request.assistant_prefix);
            return candidates;
        }

        auto base = request.assistant_prefix;
        while (!base.empty() && std::isspace(static_cast<unsigned char>(base.back()))) base.pop_back();
        if (base.size() != request.assistant_prefix.size()) append_candidate(base, request.assistant_prefix.substr(base.size()));
        constexpr auto arrow = "\xe2\x86\x92";
        if (base.size() >= std::char_traits<char>::length(arrow) &&
            base.compare(base.size() - std::char_traits<char>::length(arrow), std::char_traits<char>::length(arrow), arrow) == 0)
            append_candidate(base.substr(0, base.size() - std::char_traits<char>::length(arrow)), " ");
        append_candidate(base, " ");
        append_candidate(request.assistant_prefix);
        append_candidate(base + arrow);
        append_candidate(base + "\n");
        append_candidate(base + "[");
        append_candidate(base + "|");
        append_candidate(base + "_");
        append_candidate(std::string{"Answer:"} + arrow);
        append_candidate("Answer:\n");
        append_candidate(arrow);
        append_candidate("\n");
        append_candidate("[");
        return candidates;
    }

    bool resolve_distribution_prefix(const DistributionProbeRequest& request, const std::string& formatted_chat,
                                     std::string& formatted_prompt, std::string& assistant_prefix, std::string& label_token_prefix,
                                     std::vector<llama_token>& prompt_tokens, std::vector<int>& label_token_ids) const
    {
        for (const auto& candidate : distribution_boundary_candidates(request))
        {
            auto candidate_prompt = formatted_chat + candidate.assistant_prefix;
            std::vector<llama_token> candidate_prompt_tokens;
            if (!tokenize(candidate_prompt, true, candidate_prompt_tokens) || candidate_prompt_tokens.empty()) continue;

            std::vector<int> candidate_label_token_ids;
            candidate_label_token_ids.reserve(request.outcomes.size());
            auto valid = true;
            for (const auto& outcome : request.outcomes)
            {
                std::vector<llama_token> extended_tokens;
                if (!tokenize(candidate_prompt + candidate.label_token_prefix + outcome.label, true, extended_tokens) ||
                    extended_tokens.size() != candidate_prompt_tokens.size() + 1 ||
                    !std::equal(candidate_prompt_tokens.begin(), candidate_prompt_tokens.end(), extended_tokens.begin()))
                {
                    valid = false;
                    break;
                }
                const auto token_id = extended_tokens.back();
                if (std::find(candidate_label_token_ids.begin(), candidate_label_token_ids.end(), token_id) !=
                    candidate_label_token_ids.end())
                {
                    valid = false;
                    break;
                }
                candidate_label_token_ids.push_back(token_id);
            }
            if (!valid) continue;

            formatted_prompt = std::move(candidate_prompt);
            assistant_prefix = candidate.assistant_prefix;
            label_token_prefix = candidate.label_token_prefix;
            prompt_tokens = std::move(candidate_prompt_tokens);
            label_token_ids = std::move(candidate_label_token_ids);
            return true;
        }
        return false;
    }

    void process_distribution_probe(PendingDistributionProbe pending)
    {
        auto saved_messages = std::move(messages_);
        auto saved_response = std::move(current_response_);
        const auto saved_pending_sampling = pending_sampling_;
        const auto saved_representative_sampling = representative_sampling_;

        messages_.clear();
        current_response_.clear();
        pending_sampling_ = {};
        representative_sampling_ = {};
        previous_formatted_length_ = 0;
        if (context_ != nullptr) llama_memory_clear(llama_get_memory(context_), true);

        DistributionProbeResult result;
        result.id = pending.id;
        result.prompt = pending.request.prompt;
        result.requested_assistant_prefix = pending.request.assistant_prefix;
        result.mapping_id = pending.request.mapping_id;
        result.model_path = config_.model_path;
        result.sample_count = pending.request.sample_count;
        result.seed = pending.request.seed;
        std::vector<double> target;
        target.reserve(pending.request.outcomes.size());
        for (const auto& outcome : pending.request.outcomes) target.push_back(outcome.target_probability);
        result.task_target_entropy = DistributionMetrics::entropy(target);

        try
        {
            messages_.push_back({"user", pending.request.prompt});
            std::string formatted_chat;
            if (!format_messages(true, formatted_chat)) throw std::runtime_error("Unable to apply the model chat template");
            std::vector<llama_token> prompt_tokens;
            std::vector<int> label_token_ids;
            if (!resolve_distribution_prefix(pending.request, formatted_chat, result.formatted_prompt, result.assistant_prefix,
                                             result.label_token_prefix, prompt_tokens, label_token_ids))
                throw std::runtime_error(pending.request.auto_select_assistant_prefix
                                             ? "Unable to find a compatible one-token label boundary for this tokenizer"
                                             : "Assistant prefix \"" + pending.request.assistant_prefix +
                                                   "\" does not keep every label as one distinct token for this tokenizer");
            result.assistant_prefix_auto_selected =
                result.assistant_prefix != result.requested_assistant_prefix || !result.label_token_prefix.empty();
            result.prompt_token_count = static_cast<int>(prompt_tokens.size());

            if (prompt_tokens.size() > static_cast<std::size_t>(llama_n_ctx(context_)))
                throw std::runtime_error("Distribution prompt exceeds the model context");
            const auto batch_capacity = std::max<std::size_t>(1, static_cast<std::size_t>(llama_n_batch(context_)));
            for (std::size_t offset = 0; offset < prompt_tokens.size(); offset += batch_capacity)
            {
                if (stop_requested_.load() || reset_requested_.load() || cancel_requested_.load())
                    throw std::runtime_error("Distribution probe cancelled");
                const auto count = std::min(batch_capacity, prompt_tokens.size() - offset);
                auto batch = llama_batch_get_one(prompt_tokens.data() + offset, static_cast<int32_t>(count));
                if (llama_decode(context_, batch) != 0) throw std::runtime_error("llama_decode failed for the distribution prompt");
            }

            const auto vocabulary_size = llama_vocab_n_tokens(vocab_);
            const auto* logits = llama_get_logits_ith(context_, -1);
            if (vocabulary_size <= 0 || logits == nullptr) throw std::runtime_error("The model did not provide next-token logits");
            std::vector<llama_token_data> raw_candidates;
            raw_candidates.reserve(static_cast<std::size_t>(vocabulary_size));
            for (int token_id = 0; token_id < vocabulary_size; ++token_id) raw_candidates.push_back({token_id, logits[token_id], 0.0f});

            std::vector<std::size_t> raw_order;
            raw_order.reserve(raw_candidates.size());
            auto maximum_logit = -std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index < raw_candidates.size(); ++index)
            {
                if (!std::isfinite(raw_candidates[index].logit)) continue;
                raw_order.push_back(index);
                maximum_logit = std::max(maximum_logit, static_cast<double>(raw_candidates[index].logit));
            }
            if (raw_order.empty() || !std::isfinite(maximum_logit)) throw std::runtime_error("The model returned no finite logits");
            std::sort(raw_order.begin(), raw_order.end(),
                      [&raw_candidates](std::size_t left, std::size_t right)
                      {
                          const auto left_logit = raw_candidates[left].logit;
                          const auto right_logit = raw_candidates[right].logit;
                          return left_logit == right_logit ? left < right : left_logit > right_logit;
                      });
            double total_weight = 0.0;
            for (const auto index : raw_order) total_weight += std::exp(static_cast<double>(raw_candidates[index].logit) - maximum_logit);
            if (!(total_weight > 0.0) || !std::isfinite(total_weight)) throw std::runtime_error("The full raw softmax is invalid");

            std::vector<int> full_token_ids;
            std::vector<double> full_probabilities;
            std::vector<int> full_ranks;
            full_token_ids.reserve(raw_order.size());
            full_probabilities.reserve(raw_order.size());
            full_ranks.reserve(raw_order.size());
            for (std::size_t rank = 0; rank < raw_order.size(); ++rank)
            {
                const auto index = raw_order[rank];
                full_token_ids.push_back(raw_candidates[index].id);
                full_probabilities.push_back(std::exp(static_cast<double>(raw_candidates[index].logit) - maximum_logit) / total_weight);
                full_ranks.push_back(static_cast<int>(rank));
            }
            result.full_raw = create_distribution_stage(pending.request, label_token_ids, full_token_ids, full_probabilities, full_ranks);

            for (const auto& requested_configuration : pending.request.configurations)
            {
                if (stop_requested_.load() || reset_requested_.load() || cancel_requested_.load())
                    throw std::runtime_error("Distribution probe cancelled");
                auto candidates = raw_candidates;
                llama_token_data_array candidate_array{candidates.data(), candidates.size(), -1, false};
                CandidateTransformDiagnostics diagnostics;
                transform_candidates(&candidate_array, requested_configuration.settings, &diagnostics);
                if (diagnostics.retained_token_ids.empty()) throw std::runtime_error("A sampler configuration retained no candidates");

                DistributionConfigurationResult configuration;
                configuration.id = requested_configuration.id;
                configuration.name = requested_configuration.name;
                configuration.settings = requested_configuration.settings;
                configuration.retained_raw = create_distribution_stage(pending.request, label_token_ids, diagnostics.retained_token_ids,
                                                                       diagnostics.retained_raw_probabilities, diagnostics.retained_ranks);
                configuration.shaped = create_distribution_stage(pending.request, label_token_ids, diagnostics.retained_token_ids,
                                                                 diagnostics.shaped_probabilities, diagnostics.retained_ranks);
                create_projected_counts(configuration.shaped, pending.request.sample_count,
                                        projected_seed(pending.request.seed, pending.request.mapping_id, requested_configuration.id));
                configuration.diagnostics.support_size = static_cast<int>(diagnostics.retained_token_ids.size());
                configuration.diagnostics.retained_probability_mass = diagnostics.retained_probability_mass;
                configuration.diagnostics.sampler_raw_entropy = diagnostics.raw_metrics.entropy;
                configuration.diagnostics.sampler_target_entropy = diagnostics.target_metrics.entropy;
                configuration.diagnostics.sampler_shaped_entropy = diagnostics.shaped_metrics.entropy;
                configuration.diagnostics.sampler_entropy_error = diagnostics.shaped_metrics.entropy - diagnostics.target_metrics.entropy;
                configuration.diagnostics.raw_effective_choices = std::exp(diagnostics.raw_metrics.entropy);
                configuration.diagnostics.target_effective_choices = std::exp(diagnostics.target_metrics.entropy);
                configuration.diagnostics.shaped_effective_choices = std::exp(diagnostics.shaped_metrics.entropy);
                configuration.diagnostics.target_saturated = diagnostics.target_saturated;
                result.configurations.push_back(std::move(configuration));
            }

            result.status = "Complete";
        }
        catch (const std::exception& error)
        {
            result.status = error.what();
        }

        result.ready = true;
        result.generating = false;
        messages_ = std::move(saved_messages);
        current_response_ = std::move(saved_response);
        pending_sampling_ = saved_pending_sampling;
        representative_sampling_ = saved_representative_sampling;
        previous_formatted_length_ = 0;
        cancel_requested_.store(false);
        if (context_ != nullptr) llama_memory_clear(llama_get_memory(context_), true);
        {
            const std::lock_guard snapshot_lock(snapshot_mutex_);
            snapshot_ = std::move(pending.restoration_snapshot);
        }
        {
            const std::lock_guard distribution_lock(distribution_mutex_);
            distribution_result_ = std::move(result);
        }
        {
            const std::lock_guard offline_job_lock(offline_job_mutex_);
            offline_job_type_ = OfflineJobType::none;
        }
    }

    bool initialize_model()
    {
        if (config_.model_path.empty() || !std::filesystem::is_regular_file(config_.model_path))
        {
            update_snapshot(
                [this](SamplingSnapshot& snapshot)
                {
                    snapshot.status = "Model file not found: " + config_.model_path;
                    snapshot.model_loaded = false;
                    snapshot.generating = false;
                });
            return false;
        }

        llama_backend_init();
        backend_initialized_ = true;
        update_snapshot([](SamplingSnapshot& snapshot) { snapshot.status = "Loading model..."; });

        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = config_.gpu_layers;
        model_ = llama_model_load_from_file(config_.model_path.c_str(), model_params);
        if (model_ == nullptr)
        {
            update_snapshot([](SamplingSnapshot& snapshot) { snapshot.status = "Unable to load the GGUF model"; });
            release_model();
            return false;
        }

        vocab_ = llama_model_get_vocab(model_);
        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(std::max(512, config_.context_size));
        context_params.n_batch = 512;
        context_params.n_ubatch = 512;

        const auto hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
        const auto default_threads = std::max(1, hardware_threads > 2 ? hardware_threads - 2 : hardware_threads);
        const auto inference_threads = config_.threads > 0 ? config_.threads : default_threads;
        context_params.n_threads = inference_threads;
        context_params.n_threads_batch = inference_threads;

        context_ = llama_init_from_model(model_, context_params);
        if (context_ == nullptr)
        {
            update_snapshot([](SamplingSnapshot& snapshot) { snapshot.status = "Unable to create the llama context"; });
            release_model();
            return false;
        }

        update_snapshot(
            [](SamplingSnapshot& snapshot)
            {
                snapshot.model_loaded = true;
                snapshot.generating = false;
                snapshot.status = "Ready - enter a message";
            });
        return true;
    }

    void release_model()
    {
        if (sampler_ != nullptr)
        {
            llama_sampler_free(sampler_);
            sampler_ = nullptr;
        }
        if (context_ != nullptr)
        {
            llama_free(context_);
            context_ = nullptr;
        }
        if (model_ != nullptr)
        {
            llama_model_free(model_);
            model_ = nullptr;
        }
        vocab_ = nullptr;
        if (backend_initialized_)
        {
            llama_backend_free();
            backend_initialized_ = false;
        }
    }

    void process_message(const std::string& message, const std::string& assistant_prefix = {})
    {
        messages_.push_back({"user", message});

        std::string formatted;
        if (!format_messages(true, formatted))
        {
            messages_.pop_back();
            publish_status("Unable to apply the model chat template", false);
            return;
        }
        formatted += assistant_prefix;

        const auto first_prompt = previous_formatted_length_ == 0;
        const auto prompt_start = std::min(previous_formatted_length_, formatted.size());
        const auto prompt = formatted.substr(prompt_start);

        std::vector<llama_token> prompt_tokens;
        if (!tokenize(prompt, first_prompt, prompt_tokens) || prompt_tokens.empty())
        {
            messages_.pop_back();
            publish_status("Unable to tokenize the chat prompt", false);
            return;
        }

        const auto context_used_position = llama_memory_seq_pos_max(llama_get_memory(context_), 0);
        const auto context_used = static_cast<std::size_t>(std::max<llama_pos>(-1, context_used_position) + 1);
        const auto context_capacity = static_cast<std::size_t>(llama_n_ctx(context_));
        const auto required_before_generation = context_used + prompt_tokens.size();
        if (required_before_generation >= context_capacity)
        {
            messages_.pop_back();
            publish_status("Context is full - clear the conversation to continue", false);
            return;
        }

        const auto available_tokens = context_capacity - required_before_generation - 1;
        const auto generated_token_limit =
            static_cast<int>(std::min(static_cast<std::size_t>(std::max(1, config_.maximum_response_tokens)), available_tokens));
        if (generated_token_limit == 0)
        {
            messages_.pop_back();
            publish_status("Context is full - clear the conversation to continue", false);
            return;
        }

        if (sampler_ != nullptr) llama_sampler_free(sampler_);
        sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler_, create_sampler(this));
        llama_sampler_chain_add(sampler_, llama_sampler_init_dist(shape_settings().seed));

        current_response_.clear();
        representative_sampling_ = {};
        update_snapshot(
            [](SamplingSnapshot& snapshot)
            {
                snapshot.generating = true;
                snapshot.status = "Generating...";
                snapshot.sampling_step = 0;
                snapshot.representative_sampling = false;
                snapshot.selected_token.clear();
            });
        pending_sampling_ = {};
        publish_transcript(true);

        auto batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
        llama_token next_token = 0;
        bool has_pending_token = false;
        bool generation_cancelled = false;
        bool generation_failed = false;
        bool reached_eog = false;
        int generated_token_count = 0;
        std::string completion_status = "Ready - enter a message";

        while (generated_token_count < generated_token_limit)
        {
            if (stop_requested_.load() || reset_requested_.load()) break;
            if (cancel_requested_.load())
            {
                generation_cancelled = true;
                break;
            }

            const auto used = llama_memory_seq_pos_max(llama_get_memory(context_), 0) + 1;
            if (static_cast<std::int64_t>(used) + batch.n_tokens >= static_cast<std::int64_t>(llama_n_ctx(context_)))
            {
                completion_status = "Context limit reached - clear the conversation to continue";
                break;
            }

            if (llama_decode(context_, batch) != 0)
            {
                completion_status = "llama_decode failed - the context will rebuild on "
                                    "the next message";
                generation_failed = true;
                break;
            }

            has_pending_token = false;
            next_token = llama_sampler_sample(sampler_, context_, -1);
            if (llama_vocab_is_eog(vocab_, next_token))
            {
                pending_sampling_ = {};
                reached_eog = true;
                break;
            }

            commit_sampling_snapshot(next_token);

            const auto piece = token_to_piece(vocab_, next_token);
            current_response_ += piece;
            ++generated_token_count;
            publish_transcript(true);

            batch = llama_batch_get_one(&next_token, 1);
            has_pending_token = true;
        }

        if (stop_requested_.load() || reset_requested_.load()) return;

        if (has_pending_token && !generation_failed && llama_decode(context_, batch) != 0)
        {
            completion_status = "llama_decode failed while finalizing - the context "
                                "will rebuild on the next message";
            generation_failed = true;
        }

        cancel_requested_.store(false);
        if (generation_cancelled)
            completion_status = "Ready - generation stopped";
        else if (reached_eog)
            completion_status = "Ready - response complete";
        else if (!generation_failed && generated_token_count == generated_token_limit)
            completion_status = generated_token_limit < config_.maximum_response_tokens
                                    ? "Context limit reached - clear the conversation to continue"
                                    : "Ready - maximum response length reached";

        messages_.push_back({"assistant", assistant_prefix + current_response_});
        if (generation_failed)
        {
            llama_memory_clear(llama_get_memory(context_), true);
            previous_formatted_length_ = 0;
        }
        else
        {
            std::string completed_conversation;
            const auto decoded_conversation = formatted + current_response_;
            const auto template_matches_context = format_messages(false, completed_conversation) &&
                                                  completed_conversation.compare(0, decoded_conversation.size(), decoded_conversation) == 0;
            if (template_matches_context && append_chat_suffix(completed_conversation.substr(decoded_conversation.size())))
            {
                previous_formatted_length_ = completed_conversation.size();
            }
            else
            {
                llama_memory_clear(llama_get_memory(context_), true);
                previous_formatted_length_ = 0;
                completion_status += " - chat context will rebuild on the next turn";
            }
        }

        publish_transcript(false);
        publish_representative_sampling();
        publish_status(completion_status, false);
    }

    bool format_messages(bool add_assistant, std::string& formatted) const
    {
        if (model_ == nullptr || messages_.empty()) return false;

        std::vector<llama_chat_message> messages;
        messages.reserve(messages_.size());
        for (const auto& message : messages_) messages.push_back({message.role.c_str(), message.content.c_str()});

        const auto* chat_template = llama_model_chat_template(model_, nullptr);
        if (chat_template == nullptr) return false;

        const auto required_length = llama_chat_apply_template(chat_template, messages.data(), messages.size(), add_assistant, nullptr, 0);
        if (required_length < 0) return false;
        formatted.resize(static_cast<std::size_t>(required_length));
        const auto actual_length =
            llama_chat_apply_template(chat_template, messages.data(), messages.size(), add_assistant, formatted.data(), required_length);
        if (actual_length < 0) return false;
        formatted.resize(static_cast<std::size_t>(actual_length));
        return true;
    }

    bool tokenize(const std::string& prompt, bool add_special, std::vector<llama_token>& tokens) const
    {
        if (vocab_ == nullptr || prompt.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) return false;
        const auto required_length =
            llama_tokenize(vocab_, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, add_special, true);
        if (required_length == std::numeric_limits<int32_t>::min()) return false;
        const auto capacity = required_length < 0 ? -required_length : required_length;
        tokens.resize(static_cast<std::size_t>(capacity));
        const auto actual_length = llama_tokenize(vocab_, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                                  static_cast<int32_t>(tokens.size()), add_special, true);
        if (actual_length < 0) return false;
        tokens.resize(static_cast<std::size_t>(actual_length));
        return true;
    }

    bool append_chat_suffix(const std::string& suffix)
    {
        if (suffix.empty()) return true;

        std::vector<llama_token> tokens;
        if (!tokenize(suffix, false, tokens)) return false;
        if (tokens.empty()) return true;

        const auto context_used_position = llama_memory_seq_pos_max(llama_get_memory(context_), 0);
        const auto context_used = static_cast<std::size_t>(std::max<llama_pos>(-1, context_used_position) + 1);
        const auto context_capacity = static_cast<std::size_t>(llama_n_ctx(context_));
        if (context_used + tokens.size() > context_capacity || tokens.size() > static_cast<std::size_t>(llama_n_batch(context_)))
            return false;

        auto batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        return llama_decode(context_, batch) == 0;
    }

    static std::size_t display_rank_at(std::size_t display_index, std::size_t display_count, std::size_t candidate_count,
                                       std::size_t previous_rank)
    {
        if (candidate_count <= display_count) return display_index;

        const auto fraction = static_cast<double>(display_index) / static_cast<double>(display_count - 1);
        const auto logarithmic_rank =
            static_cast<std::size_t>(std::llround(std::expm1(fraction * std::log(static_cast<double>(candidate_count)))));
        const auto minimum_rank = display_index == 0 ? 0 : previous_rank + 1;
        const auto maximum_rank = candidate_count - (display_count - display_index);
        return std::max(minimum_rank, std::min(maximum_rank, logarithmic_rank));
    }

    void transform_candidates(llama_token_data_array* candidates, const ShapeSettings& settings, CandidateTransformDiagnostics* diagnostics)
    {
        if (candidates == nullptr || candidates->data == nullptr || candidates->size == 0) return;

        std::vector<std::size_t> ordered_indices;
        ordered_indices.reserve(candidates->size);
        float maximum_logit = -std::numeric_limits<float>::infinity();
        for (std::size_t index = 0; index < candidates->size; ++index)
        {
            const auto logit = candidates->data[index].logit;
            if (!std::isfinite(logit)) continue;
            ordered_indices.push_back(index);
            maximum_logit = std::max(maximum_logit, logit);
        }

        const auto compare_logits = [candidates](std::size_t left, std::size_t right)
        {
            const auto left_logit = candidates->data[left].logit;
            const auto right_logit = candidates->data[right].logit;
            return left_logit == right_logit ? left < right : left_logit > right_logit;
        };

        const auto candidate_cap = std::min(settings.candidate_cap, ordered_indices.size());
        if (candidate_cap == 0) return;
        std::partial_sort(ordered_indices.begin(), ordered_indices.begin() + static_cast<std::ptrdiff_t>(candidate_cap),
                          ordered_indices.end(), compare_logits);

        std::vector<float> raw_logits(candidate_cap);
        std::vector<llama_token_data> selected_candidates(candidate_cap);
        std::vector<int> retained_ranks(candidate_cap);
        for (std::size_t rank = 0; rank < candidate_cap; ++rank)
        {
            raw_logits[rank] = candidates->data[ordered_indices[rank]].logit;
            selected_candidates[rank] = candidates->data[ordered_indices[rank]];
            retained_ranks[rank] = static_cast<int>(rank);
        }

        if (settings.minimum_relative_probability > 0.0f)
        {
            const auto minimum_logit =
                raw_logits.front() + static_cast<float>(std::log(static_cast<double>(settings.minimum_relative_probability)));
            auto retained_count = std::size_t{};
            for (std::size_t rank = 0; rank < candidate_cap; ++rank)
            {
                const auto token = selected_candidates[rank].id;
                const auto protected_token = settings.protect_control_tokens && vocab_ != nullptr &&
                                             (llama_vocab_is_control(vocab_, token) || llama_vocab_is_eog(vocab_, token));
                if (raw_logits[rank] < minimum_logit && !protected_token) continue;
                raw_logits[retained_count] = raw_logits[rank];
                selected_candidates[retained_count] = selected_candidates[rank];
                retained_ranks[retained_count] = retained_ranks[rank];
                ++retained_count;
            }
            raw_logits.resize(retained_count);
            selected_candidates.resize(retained_count);
            retained_ranks.resize(retained_count);
        }
        const auto shape_count = raw_logits.size();
        if (shape_count == 0) return;

        auto shaped_logits = raw_logits;
        shape_ranked_logits(shaped_logits, settings);
        if (settings.protect_control_tokens && vocab_ != nullptr)
        {
            std::vector<bool> protected_ranks(shape_count);
            for (std::size_t rank = 0; rank < shape_count; ++rank)
            {
                const auto token = selected_candidates[rank].id;
                protected_ranks[rank] = llama_vocab_is_control(vocab_, token) || llama_vocab_is_eog(vocab_, token);
                if (protected_ranks[rank]) shaped_logits[rank] = raw_logits[rank];
            }

            std::size_t segment_begin = 0;
            auto upper_bound = std::numeric_limits<float>::infinity();
            while (segment_begin < shape_count)
            {
                auto next_protected = segment_begin;
                while (next_protected < shape_count && !protected_ranks[next_protected]) ++next_protected;
                const auto lower_bound =
                    next_protected < shape_count ? raw_logits[next_protected] : -std::numeric_limits<float>::infinity();
                for (std::size_t rank = segment_begin; rank < next_protected; ++rank)
                    shaped_logits[rank] = std::max(lower_bound, std::min(upper_bound, shaped_logits[rank]));
                if (next_protected == shape_count) break;
                upper_bound = raw_logits[next_protected];
                segment_begin = next_protected + 1;
            }
        }

        for (std::size_t rank = 0; rank < shape_count; ++rank) selected_candidates[rank].logit = shaped_logits[rank];

        ProbabilityMetrics raw_metrics;
        ProbabilityMetrics shaped_metrics;
        const auto raw_probabilities = probabilities_from_logits(raw_logits, &raw_metrics);
        const auto shaped_probabilities = probabilities_from_logits(shaped_logits, &shaped_metrics);

        ProbabilityMetrics target_metrics = raw_metrics;
        auto target_saturated = false;
        if (settings.profile != RankProfile::none && settings.diversity != 1.0f)
        {
            const auto raw_effective_choices = std::exp(static_cast<double>(raw_metrics.entropy));
            const auto requested_effective_choices =
                1.0 + static_cast<double>(settings.diversity) * std::max(0.0, raw_effective_choices - 1.0);
            const auto target_effective_choices = std::min(static_cast<double>(shape_count), requested_effective_choices);
            target_metrics.entropy = static_cast<float>(std::log(target_effective_choices));
            target_saturated = requested_effective_choices >= static_cast<double>(shape_count) - 1.0e-9;
        }

        double total_weight = 0.0;
        for (std::size_t index = 0; index < candidates->size; ++index)
        {
            const auto logit = candidates->data[index].logit;
            if (std::isfinite(logit)) total_weight += std::exp(static_cast<double>(logit - maximum_logit));
        }
        double selected_weight = 0.0;
        for (const auto logit : raw_logits) selected_weight += std::exp(static_cast<double>(logit - maximum_logit));

        if (diagnostics != nullptr)
        {
            diagnostics->retained_token_ids.reserve(shape_count);
            diagnostics->retained_raw_probabilities.reserve(shape_count);
            diagnostics->shaped_probabilities.reserve(shape_count);
            diagnostics->retained_ranks = retained_ranks;
            for (std::size_t rank = 0; rank < shape_count; ++rank)
            {
                diagnostics->retained_token_ids.push_back(selected_candidates[rank].id);
                diagnostics->retained_raw_probabilities.push_back(raw_probabilities[rank]);
                diagnostics->shaped_probabilities.push_back(shaped_probabilities[rank]);
            }
            diagnostics->retained_probability_mass = total_weight > 0.0 ? selected_weight / total_weight : 0.0;
            diagnostics->raw_metrics = raw_metrics;
            diagnostics->target_metrics = target_metrics;
            diagnostics->shaped_metrics = shaped_metrics;
            diagnostics->target_saturated = target_saturated;
        }

        std::copy(selected_candidates.begin(), selected_candidates.end(), candidates->data);
        candidates->size = shape_count;
        candidates->selected = -1;
        candidates->sorted = true;
    }

    void apply_shaping(llama_token_data_array* candidates)
    {
        CandidateTransformDiagnostics diagnostics;
        const auto settings = shape_settings();
        transform_candidates(candidates, settings, &diagnostics);
        if (diagnostics.retained_token_ids.empty()) return;

        pending_sampling_ = {};
        pending_sampling_.valid = true;
        pending_sampling_.settings = settings;
        pending_sampling_.candidate_count = diagnostics.retained_token_ids.size();
        pending_sampling_.probability_count = std::min(display_rank_count, pending_sampling_.candidate_count);
        auto previous_rank = std::size_t{};
        for (std::size_t display_index = 0; display_index < pending_sampling_.probability_count; ++display_index)
        {
            const auto rank =
                display_rank_at(display_index, pending_sampling_.probability_count, pending_sampling_.candidate_count, previous_rank);
            pending_sampling_.probability_ranks[display_index] = rank;
            pending_sampling_.raw_probabilities[display_index] = static_cast<float>(diagnostics.retained_raw_probabilities[rank]);
            pending_sampling_.shaped_probabilities[display_index] = static_cast<float>(diagnostics.shaped_probabilities[rank]);
            previous_rank = rank;
        }
        pending_sampling_.raw_entropy = diagnostics.raw_metrics.entropy;
        pending_sampling_.shaped_entropy = diagnostics.shaped_metrics.entropy;
        pending_sampling_.raw_peak_probability = diagnostics.raw_metrics.peak_probability;
        pending_sampling_.shaped_peak_probability = diagnostics.shaped_metrics.peak_probability;
        pending_sampling_.pool_probability_mass = static_cast<float>(diagnostics.retained_probability_mass);
        std::vector<float> raw_probabilities;
        std::vector<float> shaped_probabilities;
        raw_probabilities.reserve(diagnostics.retained_raw_probabilities.size());
        shaped_probabilities.reserve(diagnostics.shaped_probabilities.size());
        for (const auto probability : diagnostics.retained_raw_probabilities) raw_probabilities.push_back(static_cast<float>(probability));
        for (const auto probability : diagnostics.shaped_probabilities) shaped_probabilities.push_back(static_cast<float>(probability));
        pending_sampling_.jensen_shannon_divergence = jensen_shannon_divergence(raw_probabilities, shaped_probabilities);
    }

    void commit_sampling_snapshot(llama_token selected_token)
    {
        if (!pending_sampling_.valid) return;

        pending_sampling_.selected_token = sanitize_utf8(token_to_piece(vocab_, selected_token));
        const auto settings_match =
            representative_sampling_.valid && pending_sampling_.settings.profile == representative_sampling_.settings.profile &&
            pending_sampling_.settings.diversity == representative_sampling_.settings.diversity &&
            pending_sampling_.settings.candidate_cap == representative_sampling_.settings.candidate_cap &&
            pending_sampling_.settings.minimum_relative_probability == representative_sampling_.settings.minimum_relative_probability &&
            pending_sampling_.settings.seed == representative_sampling_.settings.seed &&
            pending_sampling_.settings.protect_control_tokens == representative_sampling_.settings.protect_control_tokens;
        if (!settings_match || pending_sampling_.raw_entropy > representative_sampling_.raw_entropy)
            representative_sampling_ = pending_sampling_;

        const std::lock_guard lock(snapshot_mutex_);
        ++snapshot_.sampling_step;
        snapshot_.representative_sampling = false;
        copy_sampling_data(pending_sampling_, snapshot_);
        pending_sampling_ = {};
    }

    static void copy_sampling_data(const PendingSamplingData& source, SamplingSnapshot& destination)
    {
        destination.selected_token = source.selected_token;
        destination.sampling_settings = source.settings;
        destination.candidate_count = source.candidate_count;
        destination.probability_count = source.probability_count;
        destination.probability_ranks = source.probability_ranks;
        destination.raw_probabilities = source.raw_probabilities;
        destination.shaped_probabilities = source.shaped_probabilities;
        destination.raw_entropy = source.raw_entropy;
        destination.shaped_entropy = source.shaped_entropy;
        destination.raw_peak_probability = source.raw_peak_probability;
        destination.shaped_peak_probability = source.shaped_peak_probability;
        destination.pool_probability_mass = source.pool_probability_mass;
        destination.jensen_shannon_divergence = source.jensen_shannon_divergence;
    }

    void publish_representative_sampling()
    {
        if (!representative_sampling_.valid) return;

        const std::lock_guard lock(snapshot_mutex_);
        snapshot_.representative_sampling = true;
        copy_sampling_data(representative_sampling_, snapshot_);
    }

    void publish_transcript(bool generating)
    {
        std::string transcript;
        for (const auto& message : messages_)
        {
            transcript += message.role == "user" ? "You: " : "Assistant: ";
            transcript += sanitize_utf8(message.content);
            transcript += "\n\n";
        }
        if (generating)
        {
            transcript += "Assistant: ";
            transcript += sanitize_utf8(current_response_);
        }

        update_snapshot(
            [generating, transcript = std::move(transcript)](SamplingSnapshot& snapshot) mutable
            {
                snapshot.generating = generating;
                snapshot.transcript = std::move(transcript);
            });
    }

    void publish_status(std::string status, bool generating)
    {
        update_snapshot(
            [status = std::move(status), generating](SamplingSnapshot& snapshot) mutable
            {
                snapshot.status = std::move(status);
                snapshot.generating = generating;
            });
    }

    void update_snapshot(const std::function<void(SamplingSnapshot&)>& update)
    {
        const std::lock_guard lock(snapshot_mutex_);
        update(snapshot_);
    }

    static const char* sampler_name(const llama_sampler*) { return "logit-scope-rank-profile"; }

    static void sampler_apply(llama_sampler* sampler, llama_token_data_array* candidates)
    {
        auto* context = static_cast<SamplerContext*>(sampler->ctx);
        if (context != nullptr && context->engine != nullptr) context->engine->apply_shaping(candidates);
    }

    static llama_sampler* sampler_clone(const llama_sampler* sampler)
    {
        const auto* context = static_cast<const SamplerContext*>(sampler->ctx);
        return context != nullptr ? llama_sampler_init(&sampler_interface(), new SamplerContext{context->engine}) : nullptr;
    }

    static void sampler_free(llama_sampler* sampler)
    {
        if (sampler != nullptr) delete static_cast<SamplerContext*>(sampler->ctx);
    }

    static llama_sampler* create_sampler(Impl* engine) { return llama_sampler_init(&sampler_interface(), new SamplerContext{engine}); }

    static llama_sampler_i& sampler_interface()
    {
        static llama_sampler_i interface = {
            &Impl::sampler_name,
            nullptr,
            &Impl::sampler_apply,
            nullptr,
            &Impl::sampler_clone,
            &Impl::sampler_free,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
        };
        return interface;
    }

    EngineConfig config_;
    std::atomic<int> settings_profile_{static_cast<int>(ShapeSettings{}.profile)};
    std::atomic<float> settings_diversity_{ShapeSettings{}.diversity};
    std::atomic<std::size_t> settings_candidate_cap_{ShapeSettings{}.candidate_cap};
    std::atomic<float> settings_minimum_relative_probability_{ShapeSettings{}.minimum_relative_probability};
    std::atomic<std::uint32_t> settings_seed_{ShapeSettings{}.seed};
    std::atomic<bool> settings_protect_control_{ShapeSettings{}.protect_control_tokens};

    std::atomic<bool> started_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> reset_requested_{false};
    std::atomic<bool> cancel_requested_{false};
    std::thread worker_;

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::deque<std::string> pending_messages_;
    std::deque<EvaluationRequest> pending_evaluations_;
    std::deque<PendingDistributionProbe> pending_distributions_;

    mutable std::mutex snapshot_mutex_;
    SamplingSnapshot snapshot_;
    mutable std::mutex preview_mutex_;
    mutable SamplingSnapshot preview_snapshot_;
    mutable ShapeSettings preview_settings_;
    mutable bool preview_valid_ = false;
    PendingSamplingData pending_sampling_;
    PendingSamplingData representative_sampling_;

    mutable std::mutex evaluation_mutex_;
    EvaluationResult evaluation_result_;
    std::atomic<std::uint64_t> next_evaluation_id_{1};

    mutable std::mutex distribution_mutex_;
    DistributionProbeResult distribution_result_;
    std::atomic<std::uint64_t> next_distribution_id_{1};

    mutable std::mutex offline_job_mutex_;
    OfflineJobType offline_job_type_ = OfflineJobType::none;

    std::vector<ChatMessage> messages_;
    std::string current_response_;
    std::size_t previous_formatted_length_ = 0;

    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    bool backend_initialized_ = false;
};

Engine::Engine(EngineConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

Engine::~Engine() = default;

void Engine::start() { impl_->start(); }

void Engine::stop() { impl_->stop(); }

bool Engine::submit_message(std::string message) { return impl_->submit_message(std::move(message)); }

std::uint64_t Engine::submit_evaluation(std::string prompt, const ShapeSettings& settings, std::string assistant_prefix)
{
    return impl_->submit_evaluation(std::move(prompt), settings, std::move(assistant_prefix));
}

std::uint64_t Engine::submit_distribution(DistributionProbeRequest request) { return impl_->submit_distribution(std::move(request)); }

void Engine::cancel_generation() { impl_->cancel_generation(); }

void Engine::clear_conversation() { impl_->clear_conversation(); }

ShapeSettings Engine::shape_settings() const { return impl_->shape_settings(); }

void Engine::set_shape_settings(const ShapeSettings& settings) { impl_->set_shape_settings(settings); }

SamplingSnapshot Engine::snapshot() const { return impl_->snapshot(); }

SamplingSnapshot Engine::preview_snapshot(const ShapeSettings& settings) const { return impl_->preview_snapshot(settings); }

EvaluationResult Engine::evaluation_result() const { return impl_->evaluation_result(); }

DistributionProbeResult Engine::distribution_result() const { return impl_->distribution_result(); }

} // namespace logit_scope
