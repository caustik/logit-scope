#include "logit_scope/rank_profile.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace
{

void require(bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
}

bool close(float left, float right, float tolerance = 1.0e-5f) { return std::abs(left - right) <= tolerance; }

logit_scope::ProbabilityMetrics probability_metrics(const std::vector<float>& logits)
{
    logit_scope::ProbabilityMetrics metrics;
    logit_scope::probabilities_from_logits(logits, &metrics);
    return metrics;
}

} // namespace

int main()
{
    using namespace logit_scope;

    std::vector<float> bypass{3.0f, 2.0f, 1.0f, -1.0f};
    const auto original = bypass;
    ShapeSettings settings;
    settings.profile = RankProfile::none;
    shape_ranked_logits(bypass, settings);
    require(bypass == original, "none profile must be exact bypass");

    for (const auto profile : {RankProfile::exponential, RankProfile::soliton, RankProfile::power, RankProfile::half_normal})
    {
        ShapeSettings identity_settings;
        identity_settings.profile = profile;
        identity_settings.diversity = 1.0f;
        auto identity = original;
        shape_ranked_logits(identity, identity_settings);
        require(identity == original, "full diversity must be exact bypass");

        ShapeSettings deterministic_settings;
        deterministic_settings.profile = profile;
        deterministic_settings.diversity = 0.0f;
        auto deterministic = original;
        shape_ranked_logits(deterministic, deterministic_settings);
        const auto deterministic_probabilities = probabilities_from_logits(deterministic);
        require(close(deterministic_probabilities.front(), 1.0f), "zero diversity must retain the top candidate");
        require(std::all_of(deterministic_probabilities.begin() + 1, deterministic_probabilities.end(),
                            [](float probability) { return probability == 0.0f; }),
                "zero diversity must suppress every lower-ranked candidate");
        require(close(deterministic.front(), original.front()), "zero diversity must preserve the top logit");

        ShapeSettings uniform_settings;
        uniform_settings.profile = profile;
        uniform_settings.diversity = 2.0f;
        const std::vector<float> sharp{12.0f, 0.0f, -1.0f, -2.0f};
        auto uniform = sharp;
        shape_ranked_logits(uniform, uniform_settings);
        require(std::all_of(uniform.begin(), uniform.end(), [&](float logit) { return close(logit, sharp.front()); }),
                "maximum diversity must reach uniform even from a low-entropy distribution");

        for (const auto candidate_count : {std::size_t{32}, std::size_t{4096}})
        {
            std::vector<float> raw_logits(candidate_count);
            for (std::size_t rank = 0; rank < raw_logits.size(); ++rank) raw_logits[rank] = 4.0f - static_cast<float>(rank) * 0.3f;
            const auto raw_metrics = probability_metrics(raw_logits);

            for (const auto diversity : {0.75f, 1.5f})
            {
                ShapeSettings shaped_settings;
                shaped_settings.profile = profile;
                shaped_settings.diversity = diversity;
                auto logits = raw_logits;
                shape_ranked_logits(logits, shaped_settings);

                const auto shaped_metrics = probability_metrics(logits);
                const auto maximum_entropy = std::log(static_cast<float>(candidate_count));
                const auto expected_entropy = diversity < 1.0f
                                                  ? raw_metrics.entropy * diversity
                                                  : raw_metrics.entropy + (maximum_entropy - raw_metrics.entropy) * (diversity - 1.0f);
                if (!close(shaped_metrics.entropy, expected_entropy, 5.0e-4f))
                    std::cerr << "entropy mismatch: profile=" << rank_profile_name(profile) << " pool=" << candidate_count
                              << " diversity=" << diversity << " actual=" << shaped_metrics.entropy << " expected=" << expected_entropy
                              << '\n';
                require(close(shaped_metrics.entropy, expected_entropy, 5.0e-4f),
                        "diversity must reach the requested entropy independent of pool size");
                require(diversity < 1.0f ? shaped_metrics.peak_probability > raw_metrics.peak_probability
                                         : shaped_metrics.peak_probability < raw_metrics.peak_probability,
                        "diversity must move the peak in the selected direction");
                require(std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }),
                        "nonzero shaping must remain finite");
                require(std::is_sorted(logits.begin(), logits.end(), std::greater<float>()), "shaping must preserve rank");
                require(close(logits.front(), 4.0f), "shaping must preserve the top logit");
            }
        }

        ShapeSettings extreme_settings;
        extreme_settings.profile = profile;
        extreme_settings.diversity = 0.05f;
        std::vector<float> extreme_logits(4096);
        for (std::size_t rank = 0; rank < extreme_logits.size(); ++rank) extreme_logits[rank] = 10.0f - static_cast<float>(rank) * 0.025f;

        shape_ranked_logits(extreme_logits, extreme_settings);
        require(std::all_of(extreme_logits.begin(), extreme_logits.end(), [](float value) { return std::isfinite(value); }),
                "extreme shaping must remain finite");
        require(std::is_sorted(extreme_logits.begin(), extreme_logits.end(), std::greater<float>()), "extreme shaping must preserve rank");

        extreme_settings.diversity = 1.95f;
        shape_ranked_logits(extreme_logits, extreme_settings);
        require(std::all_of(extreme_logits.begin(), extreme_logits.end(), [](float value) { return std::isfinite(value); }),
                "extreme loosening must remain finite");
        require(std::is_sorted(extreme_logits.begin(), extreme_logits.end(), std::greater<float>()),
                "extreme loosening must preserve rank");
    }

    ShapeSettings soliton_settings;
    soliton_settings.profile = RankProfile::soliton;
    soliton_settings.diversity = 0.5f;
    std::vector<float> soliton_logits(8, 0.0f);
    shape_ranked_logits(soliton_logits, soliton_settings);
    const auto soliton_gap = [&soliton_logits](std::size_t rank) { return soliton_logits[rank - 1] - soliton_logits[rank]; };
    require(soliton_gap(1) < soliton_gap(2) && soliton_gap(2) < soliton_gap(3), "soliton profile must have a rounded low-rank crest");
    require(soliton_gap(7) - soliton_gap(6) < soliton_gap(2) - soliton_gap(1), "soliton profile must approach exponential tail decay");

    settings.profile = RankProfile::half_normal;
    settings.diversity = 0.5f;
    shape_ranked_logits(bypass, settings);
    const auto raw = probabilities_from_logits(original);
    const auto shaped = probabilities_from_logits(bypass);
    require(jensen_shannon_divergence(raw, shaped) > 0.0f, "shaping must change distribution");
    require(jensen_shannon_divergence(raw, raw) < 1.0e-7f, "identity divergence");

    RankProfile parsed{};
    require(parse_rank_profile("none", parsed) && parsed == RankProfile::none, "none profile parsing");
    require(parse_rank_profile("soliton", parsed) && parsed == RankProfile::soliton, "soliton profile parsing");
    require(parse_rank_profile("half-normal", parsed) && parsed == RankProfile::half_normal, "profile parsing");
    require(!parse_rank_profile("uniform", parsed), "uniform profile rejection");
    require(!parse_rank_profile("normal", parsed), "invalid profile rejection");

    std::cout << "rank-profile-tests passed\n";
    return 0;
}
