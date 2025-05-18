#include "RealtimeLights.h"

#include "..\Graphics\RenderContext.h"
#include "..\Graphics\ShadowAtlas.h"
#include "..\Graphics\PerCameraLightDataConstantBuffer.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	uint32_t GetShadowSize(const LightType& type)
	{
		if (type == LightType::Spot)
			return 512;
		if (type == LightType::Directional)
			return 2048;
		if (type == LightType::Point)
			return 256;
		return 0;
	}

	uint8_t GetSliceCount(const LightType& type)
	{
		if (type == LightType::Spot)
			return 1;
		if (type == LightType::Directional)
			return 3;
		if (type == LightType::Point)
			return 6;
		return 0;
	}

	void RealtimeLights::PrepareShadows(CullingResults& results, ShadowAtlas* atlas)
	{
		for (Light* light : results.lights)
		{
			if (light->IsCastingShadows())
			{
				LightType type = light->GetType();
				atlas->Insert(light, GetShadowSize(type), GetSliceCount(type));
			}
		}
	}

	void RealtimeLights::BindLights(CullingResults& results, ShadowAtlas* atlas)
	{
		LightData mainLight = {};
		List<LightData> lights;
		for (Light* light : results.lights)
		{
			if (light->GetType() == LightType::Directional && light->IsCastingShadows())
			{
				mainLight.transform = light->GetTransform();
				mainLight.light = light;
				continue;
			}
			auto transform = light->GetTransform();
			lights.emplace_back(LightData{ transform, light });
		}
		PerCameraLightDataConstantBuffer::BindData(mainLight, lights, atlas->GetSize());
	}
}
