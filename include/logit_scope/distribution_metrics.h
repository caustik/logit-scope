#pragma once

#include <vector>

namespace logit_scope
{

struct DistributionMetricResult
{
    double total_variation = 0.0;
    double js_divergence_nats = 0.0;
    double js_distance_normalized = 0.0;
    double entropy = 0.0;
    double entropy_error = 0.0;
    double pairwise_order_accuracy = 0.0;
    double missing_target_mass = 0.0;
};

class DistributionMetrics
{
  public:
    static double entropy(const std::vector<double>& probabilities);
    static double total_variation(const std::vector<double>& lhs, const std::vector<double>& rhs);
    static double jensen_shannon_divergence(const std::vector<double>& lhs, const std::vector<double>& rhs);
    static double pairwise_order_accuracy(const std::vector<double>& predicted, const std::vector<double>& target);
    static std::vector<double> conditional_distribution(const std::vector<double>& probabilities);
    static DistributionMetricResult calculate(const std::vector<double>& predicted, const std::vector<double>& target,
                                              const std::vector<bool>& retained, double target_entropy);

  private:
    static void validate_distribution(const std::vector<double>& probabilities);
    static void validate_pair(const std::vector<double>& lhs, const std::vector<double>& rhs);
    static double sum(const std::vector<double>& values);
    static std::vector<double> normalized(const std::vector<double>& probabilities);
};

} // namespace logit_scope
