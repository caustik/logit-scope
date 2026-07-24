#pragma once

#include <string_view>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace logit_scope
{

constexpr float maximum_diversity = 2.0f;
constexpr std::size_t maximum_candidate_cap = 4096;

enum class RankProfile
{
    none,
    exponential,
    soliton,
    power,
    half_normal,
};

struct ShapeSettings
{
    RankProfile profile = RankProfile::soliton;
    float diversity = 1.88f;
    std::size_t candidate_cap = 64;
    float minimum_relative_probability = 0.01f;
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
void apply_relative_probability_floor(std::vector<float>& ranked_logits, float minimum_relative_probability);
void shape_ranked_logits(std::vector<float>& ranked_logits, const ShapeSettings& settings);
float jensen_shannon_divergence(const std::vector<float>& left, const std::vector<float>& right);

} // namespace logit_scope
