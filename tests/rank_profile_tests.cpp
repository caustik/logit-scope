#include "logit_scope/rank_profile.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
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

} // namespace

int main()
{
    using namespace logit_scope;

    for (const auto profile : {RankProfile::uniform, RankProfile::exponential, RankProfile::power, RankProfile::half_normal})
    {
        ShapeSettings settings;
        settings.profile = profile;
        settings.concentration = 0.8f;
        const auto target = target_rank_probabilities(32, settings);
        require(target.size() == 32, "target size");
        require(close(std::accumulate(target.begin(), target.end(), 0.0f), 1.0f), "target normalization");
        require(std::is_sorted(target.begin(), target.end(), std::greater<float>()), "target rank must be non-increasing");
    }

    std::vector<float> bypass{3.0f, 2.0f, 1.0f, -1.0f};
    const auto original = bypass;
    ShapeSettings settings;
    settings.blend = 0.0f;
    shape_ranked_logits(bypass, settings);
    require(bypass == original, "blend zero must be exact bypass");

    settings.profile = RankProfile::half_normal;
    settings.blend = 0.75f;
    settings.concentration = 0.6f;
    shape_ranked_logits(bypass, settings);
    require(std::is_sorted(bypass.begin(), bypass.end(), std::greater<float>()), "shaping must preserve rank");
    require(close(bypass.front(), original.front()), "top logit anchor");

    const auto raw = probabilities_from_logits(original);
    const auto shaped = probabilities_from_logits(bypass);
    require(jensen_shannon_divergence(raw, shaped) > 0.0f, "shaping must change distribution");
    require(jensen_shannon_divergence(raw, raw) < 1.0e-7f, "identity divergence");

    for (const auto profile : {RankProfile::uniform, RankProfile::exponential, RankProfile::power, RankProfile::half_normal})
    {
        ShapeSettings extreme_settings;
        extreme_settings.profile = profile;
        extreme_settings.blend = 1.0f;
        extreme_settings.concentration = 4.0f;
        std::vector<float> extreme_logits(4096);
        for (std::size_t rank = 0; rank < extreme_logits.size(); ++rank) extreme_logits[rank] = 10.0f - static_cast<float>(rank) * 0.025f;

        shape_ranked_logits(extreme_logits, extreme_settings);
        require(std::all_of(extreme_logits.begin(), extreme_logits.end(), [](float value) { return std::isfinite(value); }),
                "extreme shaping must remain finite");
        require(std::is_sorted(extreme_logits.begin(), extreme_logits.end(), std::greater<float>()), "extreme shaping must preserve rank");
    }

    RankProfile parsed{};
    require(parse_rank_profile("half-normal", parsed) && parsed == RankProfile::half_normal, "profile parsing");
    require(!parse_rank_profile("normal", parsed), "invalid profile rejection");

    std::cout << "rank-profile-tests passed\n";
    return 0;
}
