#include "logit_scope/rank_profile.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace logit_scope
{
namespace
{

constexpr int calibration_iterations = 32;
constexpr double logarithm_of_two = 0.693147180559945309417232121458176568;

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

float entropy_at_strength(const std::vector<float>& ranked_logits, RankProfile profile, double strength, bool loosen)
{
    if (ranked_logits.empty()) return 0.0f;

    double total = 0.0;
    double weighted_logit_total = 0.0;
    const auto maximum_logit = static_cast<double>(ranked_logits.front());
    auto previous_logit = maximum_logit;
    for (std::size_t rank = 0; rank < ranked_logits.size(); ++rank)
    {
        const auto rank_adjustment = strength * rank_penalty(profile, rank);
        auto shaped_logit = static_cast<double>(ranked_logits[rank]) + (loosen ? rank_adjustment : -rank_adjustment);
        if (loosen) shaped_logit = std::min(previous_logit, shaped_logit);
        previous_logit = shaped_logit;
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

void shape_ranked_logits(std::vector<float>& ranked_logits, const ShapeSettings& settings)
{
    const auto diversity = std::max(0.0f, std::min(2.0f, settings.diversity));
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
    if (diversity >= 2.0f)
    {
        std::fill(ranked_logits.begin() + 1, ranked_logits.end(), ranked_logits.front());
        return;
    }
    const auto target_entropy =
        loosen ? raw_metrics.entropy + (maximum_entropy - raw_metrics.entropy) * (diversity - 1.0f) : raw_metrics.entropy * diversity;

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
    auto previous_logit = ranked_logits.front();
    for (std::size_t rank = 1; rank < ranked_logits.size(); ++rank)
    {
        const auto rank_adjustment = strength * rank_penalty(settings.profile, rank);
        auto shaped_logit = static_cast<double>(ranked_logits[rank]) + (loosen ? rank_adjustment : -rank_adjustment);
        if (loosen) shaped_logit = std::min(static_cast<double>(previous_logit), shaped_logit);
        ranked_logits[rank] = static_cast<float>(shaped_logit);
        previous_logit = ranked_logits[rank];
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
