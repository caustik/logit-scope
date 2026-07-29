#include "logit_scope/distribution_metrics.h"

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <limits>
#include <cmath>

namespace logit_scope
{

void DistributionMetrics::validate_distribution(const std::vector<double>& probabilities)
{
    if (probabilities.empty()) throw std::invalid_argument("distribution must not be empty");
    for (const auto probability : probabilities)
    {
        if (!std::isfinite(probability) || probability < 0.0) throw std::invalid_argument("distribution contains an invalid probability");
    }
    if (!(sum(probabilities) > 0.0)) throw std::invalid_argument("distribution must contain positive mass");
}

void DistributionMetrics::validate_pair(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    validate_distribution(lhs);
    validate_distribution(rhs);
    if (lhs.size() != rhs.size()) throw std::invalid_argument("distribution sizes do not match");
}

double DistributionMetrics::sum(const std::vector<double>& values) { return std::accumulate(values.begin(), values.end(), 0.0); }

std::vector<double> DistributionMetrics::normalized(const std::vector<double>& probabilities)
{
    validate_distribution(probabilities);
    const auto total = sum(probabilities);
    std::vector<double> result;
    result.reserve(probabilities.size());
    for (const auto probability : probabilities) result.push_back(probability / total);
    return result;
}

double DistributionMetrics::entropy(const std::vector<double>& probabilities)
{
    const auto values = normalized(probabilities);
    double result = 0.0;
    for (const auto probability : values)
    {
        if (probability > 0.0) result -= probability * std::log(probability);
    }
    return result;
}

double DistributionMetrics::total_variation(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    validate_pair(lhs, rhs);
    const auto normalized_lhs = normalized(lhs);
    const auto normalized_rhs = normalized(rhs);
    double distance = 0.0;
    for (std::size_t index = 0; index < normalized_lhs.size(); ++index) distance += std::abs(normalized_lhs[index] - normalized_rhs[index]);
    return 0.5 * distance;
}

double DistributionMetrics::jensen_shannon_divergence(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    validate_pair(lhs, rhs);
    const auto normalized_lhs = normalized(lhs);
    const auto normalized_rhs = normalized(rhs);
    double divergence = 0.0;
    for (std::size_t index = 0; index < normalized_lhs.size(); ++index)
    {
        const auto midpoint = 0.5 * (normalized_lhs[index] + normalized_rhs[index]);
        if (normalized_lhs[index] > 0.0) divergence += 0.5 * normalized_lhs[index] * std::log(normalized_lhs[index] / midpoint);
        if (normalized_rhs[index] > 0.0) divergence += 0.5 * normalized_rhs[index] * std::log(normalized_rhs[index] / midpoint);
    }
    return divergence;
}

double DistributionMetrics::pairwise_order_accuracy(const std::vector<double>& predicted, const std::vector<double>& target)
{
    validate_pair(predicted, target);
    const auto normalized_predicted = normalized(predicted);
    const auto normalized_target = normalized(target);
    double score = 0.0;
    std::size_t pair_count = 0;
    for (std::size_t left = 0; left < normalized_target.size(); ++left)
    {
        for (std::size_t right = left + 1; right < normalized_target.size(); ++right)
        {
            const auto target_difference = normalized_target[left] - normalized_target[right];
            if (target_difference == 0.0) continue;
            const auto predicted_difference = normalized_predicted[left] - normalized_predicted[right];
            ++pair_count;
            if (predicted_difference == 0.0)
                score += 0.5;
            else if ((predicted_difference > 0.0) == (target_difference > 0.0))
                score += 1.0;
        }
    }
    return pair_count == 0 ? 1.0 : score / static_cast<double>(pair_count);
}

std::vector<double> DistributionMetrics::conditional_distribution(const std::vector<double>& probabilities)
{
    return normalized(probabilities);
}

DistributionMetricResult DistributionMetrics::calculate(const std::vector<double>& predicted, const std::vector<double>& target,
                                                        const std::vector<bool>& retained, double target_entropy)
{
    validate_pair(predicted, target);
    if (retained.size() != target.size()) throw std::invalid_argument("retained flags do not match distribution size");

    const auto normalized_predicted = normalized(predicted);
    const auto normalized_target = normalized(target);
    DistributionMetricResult result;
    result.total_variation = total_variation(normalized_predicted, normalized_target);
    result.js_divergence_nats = jensen_shannon_divergence(normalized_predicted, normalized_target);
    result.js_distance_normalized = std::sqrt(result.js_divergence_nats / std::log(2.0));
    result.entropy = entropy(normalized_predicted);
    result.entropy_error = result.entropy - target_entropy;
    result.pairwise_order_accuracy = pairwise_order_accuracy(normalized_predicted, normalized_target);
    for (std::size_t index = 0; index < retained.size(); ++index)
    {
        if (!retained[index]) result.missing_target_mass += normalized_target[index];
    }
    return result;
}

} // namespace logit_scope
