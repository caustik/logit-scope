#include "logit_scope/rank_profile.h"

#include <algorithm>
#include <limits>
#include <cmath>

namespace logit_scope
{
namespace
{

constexpr int calibration_iterations = 32;
constexpr double logarithm_of_two = 0.693147180559945309417232121458176568;
constexpr double maximum_gap_exponent = 64.0;

double rank_penalty(RankProfile profile, std::size_t rank)
{
    const auto rank_value = static_cast<double>(rank);
    switch (profile)
    {
    case RankProfile::exponential:
        return rank_value;
    case RankProfile::soliton:
        // Stable -log(sech²(r)) form avoids overflowing cosh for large candidate pools.
        return 2.0 * (rank_value + std::log1p(std::exp(-2.0 * rank_value)) - logarithm_of_two);
    case RankProfile::power:
        return std::log(rank_value + 1.0);
    case RankProfile::half_normal:
        return rank_value * rank_value;
    case RankProfile::none:
        return 0.0;
    }
    return 0.0;
}

double shaped_gap(double raw_gap, double profile_gap, double strength, bool loosen)
{
    if (!loosen) return raw_gap + strength * profile_gap;
    const auto exponent = std::min(maximum_gap_exponent, strength * profile_gap);
    return raw_gap * std::exp(-exponent);
}

float entropy_at_strength(const std::vector<float>& ranked_logits, RankProfile profile, double strength, bool loosen)
{
    if (ranked_logits.empty()) return 0.0f;

    double total = 0.0;
    double weighted_logit_total = 0.0;
    const auto maximum_logit = static_cast<double>(ranked_logits.front());
    auto shaped_logit = maximum_logit;
    for (std::size_t rank = 0; rank < ranked_logits.size(); ++rank)
    {
        if (rank > 0)
        {
            const auto raw_gap = std::max(0.0, static_cast<double>(ranked_logits[rank - 1]) - ranked_logits[rank]);
            const auto profile_gap = rank_penalty(profile, rank) - rank_penalty(profile, rank - 1);
            shaped_logit -= shaped_gap(raw_gap, profile_gap, strength, loosen);
        }
        const auto normalized_logit = shaped_logit - maximum_logit;
        const auto weight = std::exp(normalized_logit);
        total += weight;
        if (weight > 0.0) weighted_logit_total += weight * normalized_logit;
    }

    total = std::max(total, std::numeric_limits<double>::min());
    return static_cast<float>(std::log(total) - weighted_logit_total / total);
}

} // namespace

const char* rank_profile_name(RankProfile profile)
{
    switch (profile)
    {
    case RankProfile::none:
        return "none";
    case RankProfile::exponential:
        return "exponential";
    case RankProfile::soliton:
        return "soliton";
    case RankProfile::power:
        return "power";
    case RankProfile::half_normal:
        return "half-normal";
    }
    return "none";
}

bool parse_rank_profile(std::string_view text, RankProfile& profile)
{
    if (text == "none")
    {
        profile = RankProfile::none;
        return true;
    }
    if (text == "exponential")
    {
        profile = RankProfile::exponential;
        return true;
    }
    if (text == "soliton")
    {
        profile = RankProfile::soliton;
        return true;
    }
    if (text == "power")
    {
        profile = RankProfile::power;
        return true;
    }
    if (text == "half-normal" || text == "half_normal")
    {
        profile = RankProfile::half_normal;
        return true;
    }
    return false;
}

std::vector<float> probabilities_from_logits(const std::vector<float>& logits, ProbabilityMetrics* metrics)
{
    std::vector<float> probabilities(logits.size());
    if (logits.empty())
    {
        if (metrics != nullptr) *metrics = {};
        return probabilities;
    }

    const auto maximum = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index)
    {
        if (!std::isfinite(logits[index])) continue;
        probabilities[index] = static_cast<float>(std::exp(static_cast<double>(logits[index] - maximum)));
        total += probabilities[index];
    }

    if (total <= std::numeric_limits<double>::min())
    {
        if (metrics != nullptr) *metrics = {};
        return probabilities;
    }

    ProbabilityMetrics calculated;
    double entropy = 0.0;
    for (auto& probability : probabilities)
    {
        probability = static_cast<float>(static_cast<double>(probability) / total);
        calculated.peak_probability = std::max(calculated.peak_probability, probability);
        if (probability > 0.0f) entropy -= static_cast<double>(probability) * std::log(static_cast<double>(probability));
    }
    calculated.entropy = static_cast<float>(entropy);

    if (metrics != nullptr) *metrics = calculated;
    return probabilities;
}

void apply_relative_probability_floor(std::vector<float>& ranked_logits, float minimum_relative_probability)
{
    if (ranked_logits.size() < 2) return;

    const auto floor = std::max(0.0f, std::min(1.0f, minimum_relative_probability));
    if (floor <= 0.0f) return;

    const auto minimum_logit = ranked_logits.front() + static_cast<float>(std::log(static_cast<double>(floor)));
    const auto first_rejected =
        std::find_if(ranked_logits.begin() + 1, ranked_logits.end(), [minimum_logit](float logit) { return logit < minimum_logit; });
    ranked_logits.erase(first_rejected, ranked_logits.end());
}

void shape_ranked_logits(std::vector<float>& ranked_logits, const ShapeSettings& settings)
{
    const auto diversity = std::max(0.0f, std::min(maximum_diversity, settings.diversity));
    if (ranked_logits.size() < 2 || settings.profile == RankProfile::none || diversity == 1.0f) return;
    if (diversity <= 0.0f)
    {
        std::fill(ranked_logits.begin() + 1, ranked_logits.end(), -std::numeric_limits<float>::infinity());
        return;
    }

    ProbabilityMetrics raw_metrics;
    probabilities_from_logits(ranked_logits, &raw_metrics);
    const auto maximum_entropy = static_cast<float>(std::log(static_cast<double>(ranked_logits.size())));
    const auto loosen = diversity > 1.0f;
    const auto target_entropy =
        loosen ? std::min(maximum_entropy, raw_metrics.entropy + static_cast<float>(std::log(static_cast<double>(diversity))))
               : raw_metrics.entropy * diversity;
    if (target_entropy >= maximum_entropy)
    {
        std::fill(ranked_logits.begin() + 1, ranked_logits.end(), ranked_logits.front());
        return;
    }

    auto lower = 0.0;
    auto upper = 1.0;
    for (auto iteration = 0; iteration < calibration_iterations &&
                             (loosen ? entropy_at_strength(ranked_logits, settings.profile, upper, true) < target_entropy
                                     : entropy_at_strength(ranked_logits, settings.profile, upper, false) > target_entropy);
         ++iteration)
        upper *= 2.0;

    for (auto iteration = 0; iteration < calibration_iterations; ++iteration)
    {
        const auto middle = (lower + upper) * 0.5;
        const auto entropy = entropy_at_strength(ranked_logits, settings.profile, middle, loosen);
        if (loosen ? entropy < target_entropy : entropy > target_entropy)
            lower = middle;
        else
            upper = middle;
    }

    const auto strength = (lower + upper) * 0.5;
    auto shaped_logit = static_cast<double>(ranked_logits.front());
    auto previous_raw_logit = ranked_logits.front();
    for (std::size_t rank = 1; rank < ranked_logits.size(); ++rank)
    {
        const auto raw_logit = ranked_logits[rank];
        const auto raw_gap = std::max(0.0, static_cast<double>(previous_raw_logit) - raw_logit);
        const auto profile_gap = rank_penalty(settings.profile, rank) - rank_penalty(settings.profile, rank - 1);
        shaped_logit -= shaped_gap(raw_gap, profile_gap, strength, loosen);
        ranked_logits[rank] = static_cast<float>(std::max(static_cast<double>(std::numeric_limits<float>::lowest()), shaped_logit));
        previous_raw_logit = raw_logit;
    }
}

float jensen_shannon_divergence(const std::vector<float>& left, const std::vector<float>& right)
{
    if (left.size() != right.size() || left.empty()) return 0.0f;

    double divergence = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        const auto left_probability = std::max(0.0, static_cast<double>(left[index]));
        const auto right_probability = std::max(0.0, static_cast<double>(right[index]));
        const auto midpoint = (left_probability + right_probability) * 0.5;
        if (left_probability > 0.0 && midpoint > 0.0) divergence += 0.5 * left_probability * std::log(left_probability / midpoint);
        if (right_probability > 0.0 && midpoint > 0.0) divergence += 0.5 * right_probability * std::log(right_probability / midpoint);
    }
    return static_cast<float>(divergence);
}

} // namespace logit_scope
