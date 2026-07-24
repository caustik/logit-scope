#include "logit_scope/rank_profile.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace
{

void require(bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
}

bool close(float left, float right, float tolerance = 1.0e-5f) { return std::abs(left - right) <= tolerance; }

bool strictly_decreasing(const std::vector<float>& values, std::size_t count)
{
    count = std::min(count, values.size());
    for (std::size_t index = 1; index < count; ++index)
        if (values[index - 1] <= values[index]) return false;
    return true;
}

logit_scope::ProbabilityMetrics probability_metrics(const std::vector<float>& logits)
{
    logit_scope::ProbabilityMetrics metrics;
    logit_scope::probabilities_from_logits(logits, &metrics);
    return metrics;
}

double profile_penalty(logit_scope::RankProfile profile, std::size_t rank)
{
    const auto rank_value = static_cast<double>(rank);
    switch (profile)
    {
    case logit_scope::RankProfile::exponential:
        return rank_value;
    case logit_scope::RankProfile::soliton:
        return 2.0 * (rank_value + std::log1p(std::exp(-2.0 * rank_value)) - std::log(2.0));
    case logit_scope::RankProfile::power:
        return std::log(rank_value + 1.0);
    case logit_scope::RankProfile::half_normal:
        return rank_value * rank_value;
    case logit_scope::RankProfile::none:
    case logit_scope::RankProfile::temperature:
        return 0.0;
    }
    return 0.0;
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
    const ShapeSettings default_settings;
    require(default_settings.profile == RankProfile::soliton, "default rank profile");
    require(close(default_settings.diversity, 1.88f), "default diversity");
    require(default_settings.candidate_cap == 64, "default candidate cap");
    require(close(default_settings.minimum_relative_probability, 0.01f), "default Min-P floor");
    require(default_settings.seed == 1234, "default seed");
    require(default_settings.protect_control_tokens, "default protocol guard");
    require(close(maximum_diversity, 2.0f), "maximum diversity");

    std::vector<float> floored{0.0f, -1.0f, -2.0f, -5.0f};
    apply_relative_probability_floor(floored, 0.1f);
    require(floored.size() == 3, "Min-P floor must discard candidates below the relative peak threshold");
    std::vector<float> floor_bypass{0.0f, -1.0f, -2.0f, -5.0f};
    apply_relative_probability_floor(floor_bypass, 0.0f);
    require(floor_bypass.size() == 4, "zero Min-P floor must be an exact bypass");

    for (const auto profile :
         {RankProfile::temperature, RankProfile::exponential, RankProfile::soliton, RankProfile::power, RankProfile::half_normal})
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

        for (const auto candidate_count : {std::size_t{32}, std::size_t{4096}})
        {
            std::vector<float> raw_logits(candidate_count);
            for (std::size_t rank = 0; rank < raw_logits.size(); ++rank) raw_logits[rank] = 4.0f - static_cast<float>(rank) * 0.3f;
            const auto raw_metrics = probability_metrics(raw_logits);

            for (const auto diversity : {0.75f, 1.05f, 1.5f})
            {
                ShapeSettings shaped_settings;
                shaped_settings.profile = profile;
                shaped_settings.diversity = diversity;
                auto logits = raw_logits;
                shape_ranked_logits(logits, shaped_settings);

                const auto shaped_metrics = probability_metrics(logits);
                const auto raw_effective_choices = std::exp(raw_metrics.entropy);
                const auto expected_effective_choices =
                    std::min(static_cast<float>(candidate_count), 1.0f + diversity * (raw_effective_choices - 1.0f));
                const auto expected_entropy = std::log(expected_effective_choices);
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
                require(diversity < 1.0f || strictly_decreasing(logits, 64),
                        "loosening must not introduce equal-logit plateaus before the uniform limit");
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

        extreme_settings.diversity = 2.0f;
        shape_ranked_logits(extreme_logits, extreme_settings);
        require(std::all_of(extreme_logits.begin(), extreme_logits.end(), [](float value) { return std::isfinite(value); }),
                "extreme loosening must remain finite");
        require(std::is_sorted(extreme_logits.begin(), extreme_logits.end(), std::greater<float>()),
                "extreme loosening must preserve rank");
    }

    ShapeSettings temperature_settings;
    temperature_settings.profile = RankProfile::temperature;
    const std::vector<float> temperature_raw{5.0f, 3.0f, 0.0f, -4.0f, -9.0f};
    for (const auto diversity : {0.75f, 1.5f})
    {
        temperature_settings.diversity = diversity;
        auto temperature_shaped = temperature_raw;
        shape_ranked_logits(temperature_shaped, temperature_settings);
        const auto first_gap_ratio = (temperature_shaped[0] - temperature_shaped[1]) / (temperature_raw[0] - temperature_raw[1]);
        for (std::size_t rank = 2; rank < temperature_raw.size(); ++rank)
        {
            const auto raw_gap = temperature_raw[rank - 1] - temperature_raw[rank];
            const auto shaped_gap = temperature_shaped[rank - 1] - temperature_shaped[rank];
            require(close(shaped_gap / raw_gap, first_gap_ratio, 1.0e-4f),
                    "temperature profile must scale every adjacent logit gap equally");
        }
    }

    for (const auto profile : {RankProfile::exponential, RankProfile::soliton, RankProfile::power, RankProfile::half_normal})
    {
        std::vector<float> family_logits(16);
        for (std::size_t rank = 0; rank < family_logits.size(); ++rank)
            family_logits[rank] = static_cast<float>(3.0 - 0.7 * profile_penalty(profile, rank));

        for (const auto diversity : {0.6f, 1.6f})
        {
            ShapeSettings family_settings;
            family_settings.profile = profile;
            family_settings.diversity = diversity;
            auto family_shaped = family_logits;
            shape_ranked_logits(family_shaped, family_settings);

            const auto shaped_concentration = (static_cast<double>(family_shaped.front()) - family_shaped[1]) / profile_penalty(profile, 1);
            for (std::size_t rank = 2; rank < family_shaped.size(); ++rank)
            {
                const auto concentration =
                    (static_cast<double>(family_shaped.front()) - family_shaped[rank]) / profile_penalty(profile, rank);
                require(std::abs(concentration - shaped_concentration) < 2.0e-4,
                        "shaping a profile-family distribution must preserve that profile family");
            }
        }
    }

    ShapeSettings cliff_settings;
    cliff_settings.profile = RankProfile::exponential;
    cliff_settings.diversity = 1.5f;
    const std::vector<float> cliff_raw{0.0f, -0.1f, -3.1f};
    auto cliff_shaped = cliff_raw;
    shape_ranked_logits(cliff_shaped, cliff_settings);
    const auto small_gap_ratio = (cliff_shaped[0] - cliff_shaped[1]) / (cliff_raw[0] - cliff_raw[1]);
    const auto large_gap_ratio = (cliff_shaped[1] - cliff_shaped[2]) / (cliff_raw[1] - cliff_raw[2]);
    require(small_gap_ratio < large_gap_ratio, "loosening must soften close alternatives before crossing a large raw-logit cliff");

    ShapeSettings alternatives_settings;
    alternatives_settings.profile = RankProfile::exponential;
    alternatives_settings.diversity = 2.0f;
    const std::vector<float> low_entropy_raw{0.0f, -5.0f, -6.0f, -7.0f};
    auto alternatives_shaped = low_entropy_raw;
    shape_ranked_logits(alternatives_shaped, alternatives_settings);
    const auto low_entropy_raw_metrics = probability_metrics(low_entropy_raw);
    const auto alternatives_shaped_metrics = probability_metrics(alternatives_shaped);
    const auto raw_effective_choices = std::exp(low_entropy_raw_metrics.entropy);
    const auto shaped_effective_choices = std::exp(alternatives_shaped_metrics.entropy);
    require(close(shaped_effective_choices, 1.0f + 2.0f * (raw_effective_choices - 1.0f), 5.0e-4f),
            "diversity must scale effective alternatives rather than inventing a full choice at low entropy");

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
    require(parse_rank_profile("temperature", parsed) && parsed == RankProfile::temperature, "temperature profile parsing");
    require(parse_rank_profile("soliton", parsed) && parsed == RankProfile::soliton, "soliton profile parsing");
    require(parse_rank_profile("half-normal", parsed) && parsed == RankProfile::half_normal, "profile parsing");
    require(!parse_rank_profile("uniform", parsed), "uniform profile rejection");
    require(!parse_rank_profile("normal", parsed), "invalid profile rejection");

    std::cout << "rank-profile-tests passed\n";
    return 0;
}
