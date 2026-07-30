#include "server.h"

#include "logit_scope/engine.h"
#include "web_assets.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <memory>
#include <string>

namespace logit_scope
{
namespace
{

using json = nlohmann::json;

void append_sampling_data(json& result, const SamplingSnapshot& snapshot)
{
    result["candidateCount"] = snapshot.candidate_count;
    result["rawEntropy"] = snapshot.raw_entropy;
    result["shapedEntropy"] = snapshot.shaped_entropy;
    result["rawPeakProbability"] = snapshot.raw_peak_probability;
    result["shapedPeakProbability"] = snapshot.shaped_peak_probability;
    result["poolProbabilityMass"] = snapshot.pool_probability_mass;
    result["jensenShannonDivergence"] = snapshot.jensen_shannon_divergence;

    auto& ranks = result["probabilityRanks"];
    auto& raw = result["rawProbabilities"];
    auto& shaped = result["shapedProbabilities"];
    ranks = json::array();
    raw = json::array();
    shaped = json::array();
    for (std::size_t index = 0; index < snapshot.probability_count; ++index)
    {
        ranks.push_back(snapshot.probability_ranks[index]);
        raw.push_back(snapshot.raw_probabilities[index]);
        shaped.push_back(snapshot.shaped_probabilities[index]);
    }
}

json settings_to_json(const ShapeSettings& settings)
{
    return {
        {"profile", rank_profile_name(settings.profile)},
        {"diversity", settings.diversity},
        {"candidateCap", settings.candidate_cap},
        {"minimumRelativeProbability", settings.minimum_relative_probability},
        {"seed", settings.seed},
        {"protectControlTokens", settings.protect_control_tokens},
    };
}

ShapeSettings settings_from_json(const json& input, ShapeSettings settings)
{
    if (input.contains("profile"))
    {
        RankProfile profile;
        if (!parse_rank_profile(input.at("profile").get<std::string>(), profile)) throw std::invalid_argument("Unknown rank profile");
        settings.profile = profile;
    }
    if (input.contains("diversity")) settings.diversity = input.at("diversity").get<float>();
    if (input.contains("candidateCap")) settings.candidate_cap = input.at("candidateCap").get<std::size_t>();
    if (input.contains("minimumRelativeProbability"))
        settings.minimum_relative_probability = input.at("minimumRelativeProbability").get<float>();
    if (input.contains("seed")) settings.seed = input.at("seed").get<std::uint32_t>();
    if (input.contains("protectControlTokens")) settings.protect_control_tokens = input.at("protectControlTokens").get<bool>();
    return settings;
}

json evaluation_to_json(const EvaluationResult& result)
{
    return {
        {"id", result.id},         {"generating", result.generating},  {"ready", result.ready}, {"response", result.response},
        {"status", result.status}, {"tokenCount", result.token_count},
    };
}

json distribution_metrics_to_json(const DistributionDistanceMetrics& metrics)
{
    return {
        {"valid", metrics.valid},
        {"totalVariation", metrics.total_variation},
        {"jsDivergenceNats", metrics.js_divergence_nats},
        {"jsDistanceNormalized", metrics.js_distance_normalized},
        {"validMass", metrics.valid_mass},
        {"invalidMass", metrics.invalid_mass},
        {"conditionalEntropy", metrics.conditional_entropy},
        {"conditionalEntropyError", metrics.conditional_entropy_error},
        {"pairwiseOrderAccuracy", metrics.pairwise_order_accuracy},
        {"missingTargetMass", metrics.missing_target_mass},
    };
}

json distribution_stage_to_json(const DistributionStageResult& stage)
{
    json outcomes = json::array();
    for (const auto& outcome : stage.outcomes)
    {
        outcomes.push_back({
            {"id", outcome.id},
            {"text", outcome.text},
            {"label", outcome.label},
            {"tokenId", outcome.token_id},
            {"targetProbability", outcome.target_probability},
            {"probability", outcome.probability},
            {"retained", outcome.retained},
            {"rank", outcome.rank},
            {"projectedCount", outcome.projected_count},
        });
    }
    json top_invalid_tokens = json::array();
    for (const auto& token : stage.top_invalid_tokens)
    {
        top_invalid_tokens.push_back({
            {"tokenId", token.token_id},
            {"text", token.text},
            {"probability", token.probability},
            {"rank", token.rank},
        });
    }
    return {
        {"outcomes", std::move(outcomes)},
        {"topInvalidTokens", std::move(top_invalid_tokens)},
        {"openMetrics", distribution_metrics_to_json(stage.open_metrics)},
        {"conditionalMetrics", distribution_metrics_to_json(stage.conditional_metrics)},
        {"projectedInvalidCount", stage.projected_invalid_count},
    };
}

json distribution_to_json(const DistributionProbeResult& result)
{
    json configurations = json::array();
    for (const auto& configuration : result.configurations)
    {
        configurations.push_back({
            {"id", configuration.id},
            {"name", configuration.name},
            {"settings", settings_to_json(configuration.settings)},
            {"retainedRaw", distribution_stage_to_json(configuration.retained_raw)},
            {"shaped", distribution_stage_to_json(configuration.shaped)},
            {"diagnostics",
             {
                 {"supportSize", configuration.diagnostics.support_size},
                 {"retainedProbabilityMass", configuration.diagnostics.retained_probability_mass},
                 {"samplerRawEntropy", configuration.diagnostics.sampler_raw_entropy},
                 {"samplerTargetEntropy", configuration.diagnostics.sampler_target_entropy},
                 {"samplerShapedEntropy", configuration.diagnostics.sampler_shaped_entropy},
                 {"samplerEntropyError", configuration.diagnostics.sampler_entropy_error},
                 {"rawEffectiveChoices", configuration.diagnostics.raw_effective_choices},
                 {"targetEffectiveChoices", configuration.diagnostics.target_effective_choices},
                 {"shapedEffectiveChoices", configuration.diagnostics.shaped_effective_choices},
                 {"targetSaturated", configuration.diagnostics.target_saturated},
             }},
        });
    }
    return {
        {"id", result.id},
        {"generating", result.generating},
        {"ready", result.ready},
        {"status", result.status},
        {"prompt", result.prompt},
        {"formattedPrompt", result.formatted_prompt},
        {"requestedAssistantPrefix", result.requested_assistant_prefix},
        {"assistantPrefix", result.assistant_prefix},
        {"labelTokenPrefix", result.label_token_prefix},
        {"assistantPrefixAutoSelected", result.assistant_prefix_auto_selected},
        {"mappingId", result.mapping_id},
        {"modelPath", result.model_path},
        {"promptTokenCount", result.prompt_token_count},
        {"sampleCount", result.sample_count},
        {"seed", result.seed},
        {"taskTargetEntropy", result.task_target_entropy},
        {"fullRaw", distribution_stage_to_json(result.full_raw)},
        {"configurations", std::move(configurations)},
    };
}

DistributionProbeRequest distribution_from_json(const json& input, const ShapeSettings& base_settings)
{
    DistributionProbeRequest request;
    request.prompt = input.value("prompt", std::string{});
    request.assistant_prefix = input.value("assistantPrefix", std::string{"Answer: "});
    request.auto_select_assistant_prefix = input.value("autoSelectAssistantPrefix", true);
    request.mapping_id = input.value("mappingId", std::string{});
    request.sample_count = input.value("sampleCount", std::size_t{10000});
    request.seed = input.value("seed", std::uint32_t{1234});
    for (const auto& input_outcome : input.at("outcomes"))
    {
        DistributionOutcomeRequest outcome;
        outcome.id = input_outcome.value("id", std::string{});
        outcome.text = input_outcome.value("text", outcome.id);
        outcome.label = input_outcome.value("label", std::string{});
        outcome.target_probability = input_outcome.at("targetProbability").get<double>();
        request.outcomes.push_back(std::move(outcome));
    }
    for (const auto& input_configuration : input.at("configurations"))
    {
        DistributionConfigurationRequest configuration;
        configuration.id = input_configuration.value("id", std::string{});
        configuration.name = input_configuration.value("name", std::string{});
        configuration.settings = settings_from_json(input_configuration.value("settings", json::object()), base_settings);
        request.configurations.push_back(std::move(configuration));
    }
    return request;
}

json snapshot_to_json(const SamplingSnapshot& snapshot, const SamplingSnapshot& preview, const ShapeSettings& settings)
{
    json result = {
        {"modelLoaded", snapshot.model_loaded},
        {"generating", snapshot.generating},
        {"status", snapshot.status},
        {"transcript", snapshot.transcript},
        {"selectedToken", snapshot.selected_token},
        {"samplingStep", snapshot.sampling_step},
        {"representativeSampling", snapshot.representative_sampling},
        {"settings", settings_to_json(settings)},
        {"samplingSettings", settings_to_json(snapshot.sampling_settings)},
    };
    append_sampling_data(result, snapshot);
    result["preview"] = json::object();
    append_sampling_data(result["preview"], preview);
    return result;
}

void send_json(httplib::Response& response, const json& value, int status = 200)
{
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_content(value.dump(), "application/json; charset=utf-8");
}

} // namespace

class Server::Impl
{
  public:
    explicit Impl(Engine& engine) : engine_(engine)
    {
        server_.set_payload_max_length(1024 * 1024);

        server_.Get("/", [](const httplib::Request&, httplib::Response& response)
                    { response.set_content(std::string(web_assets::index_html), "text/html; charset=utf-8"); });
        server_.Get("/app.js", [](const httplib::Request&, httplib::Response& response)
                    { response.set_content(std::string(web_assets::app_js), "text/javascript; charset=utf-8"); });
        server_.Get("/style.css", [](const httplib::Request&, httplib::Response& response)
                    { response.set_content(std::string(web_assets::style_css), "text/css; charset=utf-8"); });

        server_.Get("/api/snapshot",
                    [this](const httplib::Request&, httplib::Response& response)
                    {
                        const auto settings = engine_.shape_settings();
                        send_json(response, snapshot_to_json(engine_.snapshot(), engine_.preview_snapshot(settings), settings));
                    });

        server_.Post("/api/message",
                     [this](const httplib::Request& request, httplib::Response& response)
                     {
                         try
                         {
                             const auto input = json::parse(request.body);
                             const auto message = input.value("message", std::string{});
                             if (message.empty())
                             {
                                 send_json(response, {{"error", "Message is empty"}}, 400);
                                 return;
                             }
                             if (!engine_.submit_message(message))
                             {
                                 send_json(response, {{"error", "The model is not ready for another message"}}, 409);
                                 return;
                             }
                             send_json(response, {{"accepted", true}}, 202);
                         }
                         catch (const std::exception& error)
                         {
                             send_json(response, {{"error", error.what()}}, 400);
                         }
                     });

        server_.Post("/api/settings",
                     [this](const httplib::Request& request, httplib::Response& response)
                     {
                         try
                         {
                             const auto input = json::parse(request.body);
                             engine_.set_shape_settings(settings_from_json(input, engine_.shape_settings()));
                             send_json(response, {{"accepted", true}});
                         }
                         catch (const std::exception& error)
                         {
                             send_json(response, {{"error", error.what()}}, 400);
                         }
                     });

        server_.Post("/api/evaluation",
                     [this](const httplib::Request& request, httplib::Response& response)
                     {
                         try
                         {
                             const auto input = json::parse(request.body);
                             const auto prompt = input.value("prompt", std::string{});
                             const auto assistant_prefix = input.value("assistantPrefix", std::string{});
                             const auto settings = settings_from_json(input.value("settings", json::object()), engine_.shape_settings());
                             const auto id = engine_.submit_evaluation(prompt, settings, assistant_prefix);
                             if (id == 0)
                             {
                                 send_json(response, {{"error", "The model is not ready for an evaluation response"}}, 409);
                                 return;
                             }
                             send_json(response, {{"accepted", true}, {"id", id}}, 202);
                         }
                         catch (const std::exception& error)
                         {
                             send_json(response, {{"error", error.what()}}, 400);
                         }
                     });

        server_.Get("/api/evaluation", [this](const httplib::Request&, httplib::Response& response)
                    { send_json(response, evaluation_to_json(engine_.evaluation_result())); });

        server_.Post("/api/distribution",
                     [this](const httplib::Request& request, httplib::Response& response)
                     {
                         try
                         {
                             const auto input = json::parse(request.body);
                             const auto id = engine_.submit_distribution(distribution_from_json(input, engine_.shape_settings()));
                             if (id == 0)
                             {
                                 send_json(response, {{"error", "The model is not ready for a distribution probe"}}, 409);
                                 return;
                             }
                             send_json(response, {{"accepted", true}, {"id", id}}, 202);
                         }
                         catch (const std::exception& error)
                         {
                             send_json(response, {{"error", error.what()}}, 400);
                         }
                     });

        server_.Get("/api/distribution", [this](const httplib::Request&, httplib::Response& response)
                    { send_json(response, distribution_to_json(engine_.distribution_result())); });

        server_.Post("/api/stop",
                     [this](const httplib::Request&, httplib::Response& response)
                     {
                         engine_.cancel_generation();
                         send_json(response, {{"accepted", true}}, 202);
                     });

        server_.Post("/api/clear",
                     [this](const httplib::Request&, httplib::Response& response)
                     {
                         engine_.clear_conversation();
                         send_json(response, {{"accepted", true}}, 202);
                     });

        server_.set_error_handler(
            [](const httplib::Request&, httplib::Response& response)
            {
                if (!response.body.empty()) return;
                const auto status = response.status;
                send_json(response, {{"error", status == 404 ? "Not found" : "Request failed"}}, status);
            });
    }

    bool listen(const std::string& host, int port) { return server_.listen(host, port); }

    void stop() { server_.stop(); }

  private:
    Engine& engine_;
    httplib::Server server_;
};

Server::Server(Engine& engine) : impl_(new Impl(engine)) {}

Server::~Server() { delete impl_; }

bool Server::listen(const std::string& host, int port) { return impl_->listen(host, port); }

void Server::stop() { impl_->stop(); }

} // namespace logit_scope
