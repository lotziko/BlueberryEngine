#include "SamplerHelper.h"

namespace Blueberry
{
    struct SamplerDefinition
    {
        size_t nameHash;
        FilterMode filterMode;
        WrapMode wrapMode;
    };

    bool SamplerHelper::ParseName(size_t nameHash, FilterMode& filterMode, WrapMode& wrapMode)
    {
        static const SamplerDefinition samplers[] =
        {
            { TO_HASH("_PointClamp_Sampler"), FilterMode::Point, WrapMode::Clamp },
            { TO_HASH("_PointRepeat_Sampler"), FilterMode::Point, WrapMode::Repeat },
            { TO_HASH("_LinearClamp_Sampler"), FilterMode::Bilinear, WrapMode::Clamp },
            { TO_HASH("_LinearRepeat_Sampler"), FilterMode::Bilinear, WrapMode::Repeat },
        };
        
        for (const auto& sampler : samplers)
        {
            if (sampler.nameHash == nameHash)
            {
                filterMode = sampler.filterMode;
                wrapMode = sampler.wrapMode;
                return true;
            }
        }
        return false;
    }
}