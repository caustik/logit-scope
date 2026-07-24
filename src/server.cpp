#include "server.h"

#include "logit_scope/engine.h"
#include "web_assets.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
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
                             auto settings = engine_.shape_settings();
                             if (input.contains("profile"))
                             {
                                 RankProfile profile;
                                 if (!parse_rank_profile(input.at("profile").get<std::string>(), profile))
                                 {
                                     send_json(response, {{"error", "Unknown rank profile"}}, 400);
                                     return;
                                 }
                                 settings.profile = profile;
                             }
                             if (input.contains("diversity")) settings.diversity = input.at("diversity").get<float>();
                             if (input.contains("candidateCap")) settings.candidate_cap = input.at("candidateCap").get<std::size_t>();
                             if (input.contains("minimumRelativeProbability"))
                                 settings.minimum_relative_probability = input.at("minimumRelativeProbability").get<float>();
                             if (input.contains("seed")) settings.seed = input.at("seed").get<std::uint32_t>();
                             if (input.contains("protectControlTokens"))
                                 settings.protect_control_tokens = input.at("protectControlTokens").get<bool>();
                             engine_.set_shape_settings(settings);
                             send_json(response, {{"accepted", true}});
                         }
                         catch (const std::exception& error)
                         {
                             send_json(response, {{"error", error.what()}}, 400);
                         }
                     });

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

        server_.set_error_handler([](const httplib::Request&, httplib::Response& response)
                                  { send_json(response, {{"error", "Not found"}}, 404); });
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
