#include "logit_scope/distribution_metrics.h"

#include <stdexcept>
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_near(double actual, double expected, double tolerance, const char* message)
{
    if (std::abs(actual - expected) > tolerance)
    {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error(message);
    }
}

template <typename Callback> void require_throws(Callback callback, const char* message)
{
    try
    {
        callback();
    }
    catch (const std::invalid_argument&)
    {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main()
{
    using logit_scope::DistributionMetrics;

    require_near(DistributionMetrics::total_variation({0.5, 0.5}, {0.5, 0.5}), 0.0, 1.0e-12, "identical TV");
    require_near(DistributionMetrics::total_variation({1.0, 0.0}, {0.0, 1.0}), 1.0, 1.0e-12, "disjoint TV");
    require_near(DistributionMetrics::jensen_shannon_divergence({1.0, 0.0}, {0.0, 1.0}), std::log(2.0), 1.0e-12, "disjoint JSD");
    require_near(DistributionMetrics::entropy({0.25, 0.25, 0.25, 0.25}), std::log(4.0), 1.0e-12, "uniform entropy");
    require_near(DistributionMetrics::pairwise_order_accuracy({0.6, 0.3, 0.1}, {0.5, 0.4, 0.1}), 1.0, 1.0e-12, "correct ordering");
    require_near(DistributionMetrics::pairwise_order_accuracy({0.1, 0.3, 0.6}, {0.5, 0.4, 0.1}), 0.0, 1.0e-12, "reversed ordering");
    require_near(DistributionMetrics::pairwise_order_accuracy({0.5, 0.5, 0.1}, {0.6, 0.3, 0.1}), 5.0 / 6.0, 1.0e-12,
                 "predicted ties receive half credit");
    require_near(DistributionMetrics::pairwise_order_accuracy({0.1, 0.2}, {0.5, 0.5}), 1.0, 1.0e-12, "target ties are excluded");

    const auto conditional = DistributionMetrics::conditional_distribution({2.0, 3.0, 5.0});
    require_near(conditional[0], 0.2, 1.0e-12, "conditional normalization");
    require_near(conditional[2], 0.5, 1.0e-12, "conditional normalization tail");

    const auto metrics = DistributionMetrics::calculate({0.6, 0.3, 0.1}, {0.5, 0.4, 0.1}, {true, false, true},
                                                        DistributionMetrics::entropy({0.5, 0.4, 0.1}));
    require_near(metrics.missing_target_mass, 0.4, 1.0e-12, "missing target mass");

    require_throws([] { DistributionMetrics::entropy({}); }, "empty distributions must fail");
    require_throws([] { DistributionMetrics::entropy({-0.1, 1.1}); }, "negative probabilities must fail");
    require_throws([] { DistributionMetrics::entropy({std::numeric_limits<double>::infinity(), 1.0}); },
                   "non-finite probabilities must fail");
    require_throws([] { DistributionMetrics::total_variation({1.0}, {0.5, 0.5}); }, "size mismatch must fail");
    require_throws([] { DistributionMetrics::conditional_distribution({0.0, 0.0}); }, "zero mass must fail");

    std::cout << "distribution-metrics-tests passed\n";
    return 0;
}
