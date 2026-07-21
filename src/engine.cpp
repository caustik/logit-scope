#include "logit_scope/engine.h"

#include <llama.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

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
            const std::lock_guard snapshot_lock(snapshot_mutex_);
            if (!snapshot_.model_loaded || snapshot_.generating) return false;
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
        settings.candidate_count = settings_candidate_count_.load();
        settings.seed = settings_seed_.load();
        settings.protect_control_tokens = settings_protect_control_.load();
        return settings;
    }

    void set_shape_settings(const ShapeSettings& settings)
    {
        const auto candidate_count = std::max<std::size_t>(2, std::min<std::size_t>(4096, settings.candidate_count));
        settings_profile_.store(static_cast<int>(settings.profile));
        settings_diversity_.store(std::max(0.0f, std::min(2.0f, settings.diversity)));
        settings_candidate_count_.store(candidate_count);
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
            settings.candidate_count == preview_settings_.candidate_count)
            return preview_snapshot_;

        SamplingSnapshot preview;
        preview.candidate_count = std::max<std::size_t>(2, std::min<std::size_t>(4096, settings.candidate_count));

        std::vector<float> raw_logits(preview.candidate_count);
        for (std::size_t rank = 0; rank < raw_logits.size(); ++rank)
            raw_logits[rank] = static_cast<float>(-1.2 * std::log(static_cast<double>(rank) + 1.0));
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

  private:
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

    void run()
    {
        if (!initialize_model()) return;

        while (!stop_requested_.load())
        {
            std::string message;
            {
                std::unique_lock lock(queue_mutex_);
                queue_condition_.wait(lock,
                                      [this] { return stop_requested_.load() || reset_requested_.load() || !pending_messages_.empty(); });
                if (stop_requested_.load()) break;

                if (reset_requested_.exchange(false))
                {
                    pending_messages_.clear();
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
            }

            if (!message.empty()) process_message(message);
        }

        release_model();
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

    void process_message(const std::string& message)
    {
        messages_.push_back({"user", message});

        std::string formatted;
        if (!format_messages(true, formatted))
        {
            messages_.pop_back();
            publish_status("Unable to apply the model chat template", false);
            return;
        }

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

        messages_.push_back({"assistant", current_response_});
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

    void apply_shaping(llama_token_data_array* candidates)
    {
        if (candidates == nullptr || candidates->data == nullptr || candidates->size == 0) return;

        const auto settings = shape_settings();
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

        const auto shape_count = std::min(settings.candidate_count, ordered_indices.size());
        if (shape_count == 0) return;
        std::partial_sort(ordered_indices.begin(), ordered_indices.begin() + static_cast<std::ptrdiff_t>(shape_count),
                          ordered_indices.end(), compare_logits);

        std::vector<float> raw_logits(shape_count);
        std::vector<llama_token_data> selected_candidates(shape_count);
        for (std::size_t rank = 0; rank < shape_count; ++rank)
        {
            raw_logits[rank] = candidates->data[ordered_indices[rank]].logit;
            selected_candidates[rank] = candidates->data[ordered_indices[rank]];
        }

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

        double total_weight = 0.0;
        for (std::size_t index = 0; index < candidates->size; ++index)
        {
            const auto logit = candidates->data[index].logit;
            if (std::isfinite(logit)) total_weight += std::exp(static_cast<double>(logit - maximum_logit));
        }
        double selected_weight = 0.0;
        for (const auto logit : raw_logits) selected_weight += std::exp(static_cast<double>(logit - maximum_logit));

        pending_sampling_ = {};
        pending_sampling_.valid = true;
        pending_sampling_.settings = settings;
        pending_sampling_.candidate_count = shape_count;
        pending_sampling_.probability_count = std::min(display_rank_count, shape_count);
        auto previous_rank = std::size_t{};
        for (std::size_t display_index = 0; display_index < pending_sampling_.probability_count; ++display_index)
        {
            const auto rank = display_rank_at(display_index, pending_sampling_.probability_count, shape_count, previous_rank);
            pending_sampling_.probability_ranks[display_index] = rank;
            pending_sampling_.raw_probabilities[display_index] = raw_probabilities[rank];
            pending_sampling_.shaped_probabilities[display_index] = shaped_probabilities[rank];
            previous_rank = rank;
        }
        pending_sampling_.raw_entropy = raw_metrics.entropy;
        pending_sampling_.shaped_entropy = shaped_metrics.entropy;
        pending_sampling_.raw_peak_probability = raw_metrics.peak_probability;
        pending_sampling_.shaped_peak_probability = shaped_metrics.peak_probability;
        pending_sampling_.pool_probability_mass = total_weight > 0.0 ? static_cast<float>(selected_weight / total_weight) : 0.0f;
        pending_sampling_.jensen_shannon_divergence = jensen_shannon_divergence(raw_probabilities, shaped_probabilities);

        std::copy(selected_candidates.begin(), selected_candidates.end(), candidates->data);
        candidates->size = shape_count;
        candidates->selected = -1;
        candidates->sorted = true;
    }

    void commit_sampling_snapshot(llama_token selected_token)
    {
        if (!pending_sampling_.valid) return;

        pending_sampling_.selected_token = sanitize_utf8(token_to_piece(vocab_, selected_token));
        const auto settings_match =
            representative_sampling_.valid && pending_sampling_.settings.profile == representative_sampling_.settings.profile &&
            pending_sampling_.settings.diversity == representative_sampling_.settings.diversity &&
            pending_sampling_.settings.candidate_count == representative_sampling_.settings.candidate_count &&
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
    std::atomic<std::size_t> settings_candidate_count_{ShapeSettings{}.candidate_count};
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

    mutable std::mutex snapshot_mutex_;
    SamplingSnapshot snapshot_;
    mutable std::mutex preview_mutex_;
    mutable SamplingSnapshot preview_snapshot_;
    mutable ShapeSettings preview_settings_;
    mutable bool preview_valid_ = false;
    PendingSamplingData pending_sampling_;
    PendingSamplingData representative_sampling_;

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

void Engine::cancel_generation() { impl_->cancel_generation(); }

void Engine::clear_conversation() { impl_->clear_conversation(); }

ShapeSettings Engine::shape_settings() const { return impl_->shape_settings(); }

void Engine::set_shape_settings(const ShapeSettings& settings) { impl_->set_shape_settings(settings); }

SamplingSnapshot Engine::snapshot() const { return impl_->snapshot(); }

SamplingSnapshot Engine::preview_snapshot(const ShapeSettings& settings) const { return impl_->preview_snapshot(settings); }

} // namespace logit_scope
