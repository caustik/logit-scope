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

json snapshot_to_json(const SamplingSnapshot& snapshot, const ShapeSettings& settings)
{
    json result = {
        {"modelLoaded", snapshot.model_loaded},
        {"generating", snapshot.generating},
        {"status", snapshot.status},
        {"transcript", snapshot.transcript},
        {"selectedToken", snapshot.selected_token},
        {"samplingStep", snapshot.sampling_step},
        {"candidateCount", snapshot.candidate_count},
        {"rawEntropy", snapshot.raw_entropy},
        {"shapedEntropy", snapshot.shaped_entropy},
        {"rawPeakProbability", snapshot.raw_peak_probability},
        {"shapedPeakProbability", snapshot.shaped_peak_probability},
        {"poolProbabilityMass", snapshot.pool_probability_mass},
        {"jensenShannonDivergence", snapshot.jensen_shannon_divergence},
        {"settings",
         {
             {"profile", rank_profile_name(settings.profile)},
             {"blend", settings.blend},
             {"concentration", settings.concentration},
             {"candidateCount", settings.candidate_count},
             {"seed", settings.seed},
             {"protectControlTokens", settings.protect_control_tokens},
         }},
    };

    auto& raw = result["rawProbabilities"];
    auto& target = result["targetProbabilities"];
    auto& shaped = result["shapedProbabilities"];
    raw = json::array();
    target = json::array();
    shaped = json::array();
    for (std::size_t index = 0; index < snapshot.probability_count; ++index)
    {
        raw.push_back(snapshot.raw_probabilities[index]);
        target.push_back(snapshot.target_probabilities[index]);
        shaped.push_back(snapshot.shaped_probabilities[index]);
    }
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

        server_.Get("/api/snapshot", [this](const httplib::Request&, httplib::Response& response)
                    { send_json(response, snapshot_to_json(engine_.snapshot(), engine_.shape_settings())); });

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
                             if (input.contains("blend")) settings.blend = input.at("blend").get<float>();
                             if (input.contains("concentration")) settings.concentration = input.at("concentration").get<float>();
                             if (input.contains("candidateCount")) settings.candidate_count = input.at("candidateCount").get<std::size_t>();
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
