#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace logit_scope
{

enum class RankProfile
{
    uniform,
    exponential,
    power,
    half_normal,
};

struct ShapeSettings
{
    RankProfile profile = RankProfile::power;
    float blend = 1.0f;
    float concentration = 2.8f;
    std::size_t candidate_count = 4096;
    std::uint32_t seed = 1234;
    bool protect_control_tokens = true;
};

struct ProbabilityMetrics
{
    float entropy = 0.0f;
    float peak_probability = 0.0f;
};

const char* rank_profile_name(RankProfile profile);
bool parse_rank_profile(std::string_view text, RankProfile& profile);

std::vector<float> probabilities_from_logits(const std::vector<float>& logits, ProbabilityMetrics* metrics = nullptr);
std::vector<float> target_rank_probabilities(std::size_t candidate_count, const ShapeSettings& settings);
void shape_ranked_logits(std::vector<float>& ranked_logits, const ShapeSettings& settings);
float jensen_shannon_divergence(const std::vector<float>& left, const std::vector<float>& right);

} // namespace logit_scope
