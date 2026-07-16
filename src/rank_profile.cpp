#include "logit_scope/rank_profile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace logit_scope
{
namespace
{

constexpr float minimum_concentration = 0.05f;
constexpr float maximum_concentration = 4.0f;

float clamp(float value, float minimum, float maximum) { return std::max(minimum, std::min(maximum, value)); }

double safe_probability(float value) { return std::max(static_cast<double>(value), std::numeric_limits<double>::min()); }

} // namespace

const char* rank_profile_name(RankProfile profile)
{
    switch (profile)
    {
    case RankProfile::uniform:
        return "uniform";
    case RankProfile::power:
        return "power";
    case RankProfile::half_normal:
        return "half-normal";
    case RankProfile::exponential:
    default:
        return "exponential";
    }
}

bool parse_rank_profile(std::string_view text, RankProfile& profile)
{
    if (text == "uniform")
    {
        profile = RankProfile::uniform;
        return true;
    }
    if (text == "exponential")
    {
        profile = RankProfile::exponential;
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
    for (auto& probability : probabilities)
    {
        probability = static_cast<float>(static_cast<double>(probability) / total);
        calculated.peak_probability = std::max(calculated.peak_probability, probability);
        if (probability > 0.0f) calculated.entropy -= probability * static_cast<float>(std::log(probability));
    }

    if (metrics != nullptr) *metrics = calculated;
    return probabilities;
}

std::vector<float> target_rank_probabilities(std::size_t candidate_count, const ShapeSettings& settings)
{
    std::vector<float> probabilities(candidate_count);
    if (candidate_count == 0) return probabilities;

    const auto concentration = static_cast<double>(clamp(settings.concentration, minimum_concentration, maximum_concentration));
    double total = 0.0;
    for (std::size_t rank = 0; rank < candidate_count; ++rank)
    {
        const auto rank_value = static_cast<double>(rank);
        double weight = 1.0;
        switch (settings.profile)
        {
        case RankProfile::exponential:
            weight = std::exp(-concentration * rank_value);
            break;
        case RankProfile::power:
            weight = std::pow(rank_value + 1.0, -concentration);
            break;
        case RankProfile::half_normal:
            weight = std::exp(-0.5 * concentration * concentration * rank_value * rank_value);
            break;
        case RankProfile::uniform:
        default:
            break;
        }

        probabilities[rank] = static_cast<float>(std::max(weight, std::numeric_limits<double>::min()));
        total += probabilities[rank];
    }

    total = std::max(total, std::numeric_limits<double>::min());
    for (auto& probability : probabilities) probability = static_cast<float>(static_cast<double>(probability) / total);
    return probabilities;
}

void shape_ranked_logits(std::vector<float>& ranked_logits, const ShapeSettings& settings)
{
    const auto blend = clamp(settings.blend, 0.0f, 1.0f);
    if (ranked_logits.size() < 2 || blend <= 0.0f) return;

    const auto raw_probabilities = probabilities_from_logits(ranked_logits);
    const auto target_probabilities = target_rank_probabilities(ranked_logits.size(), settings);
    if (raw_probabilities.size() != target_probabilities.size() || raw_probabilities.empty()) return;

    const auto raw_log_top = std::log(safe_probability(raw_probabilities.front()));
    const auto target_log_top = std::log(safe_probability(target_probabilities.front()));
    const auto blended_log_top = raw_log_top + static_cast<double>(blend) * (target_log_top - raw_log_top);
    const auto log_normalizer = static_cast<double>(ranked_logits.front()) - blended_log_top;

    for (std::size_t rank = 0; rank < ranked_logits.size(); ++rank)
    {
        const auto raw_log_probability = std::log(safe_probability(raw_probabilities[rank]));
        const auto target_log_probability = std::log(safe_probability(target_probabilities[rank]));
        ranked_logits[rank] = static_cast<float>(
            raw_log_probability + static_cast<double>(blend) * (target_log_probability - raw_log_probability) + log_normalizer);
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
