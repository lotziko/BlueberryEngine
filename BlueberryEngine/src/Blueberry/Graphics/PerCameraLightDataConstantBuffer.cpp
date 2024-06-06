#include "bbpch.h"
#include "PerCameraLightDataConstantBuffer.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"

#include "GfxDevice.h"
#include "GfxBuffer.h"

namespace Blueberry
{
	#define MAX_REALTIME_LIGHTS 128

	struct CONSTANTS
	{
		Vector4 lightsCount;
		Vector4 lightPosition[MAX_REALTIME_LIGHTS];
		Vector4 lightColor[MAX_REALTIME_LIGHTS];
		Vector4 lightAttenuation[MAX_REALTIME_LIGHTS];
		Vector4 lightDirection[MAX_REALTIME_LIGHTS];
	};

	Vector4 GetAttenuation(float lightRange)
	{
		float lightRangeSqr = lightRange * lightRange;
		float fadeStartDistanceSqr = 0.8f * 0.8f * lightRangeSqr;
		float fadeRangeSqr = (fadeStartDistanceSqr - lightRangeSqr);
		float lightRangeSqrOverFadeRangeSqr = -lightRangeSqr / fadeRangeSqr;
		float oneOverLightRangeSqr = 1.0f / Max(0.0001f, lightRangeSqr);
		return Vector4(oneOverLightRangeSqr, lightRangeSqrOverFadeRangeSqr, 0, 0);
	}

	void PerCameraLightDataConstantBuffer::BindData(const std::vector<LightData> lights)
	{
		static size_t perCameraLightDataId = TO_HASH("PerCameraLightData");

		if (s_ConstantBuffer == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		CONSTANTS constants;
		constants.lightsCount = Vector4(lights.size(), 0.0f, 0.0f, 0.0f);
		for (int i = 0; i < lights.size(); i++)
		{
			LightData data = lights[i];
			Vector3 position = data.transform->GetPosition();
			Color color = data.light->GetColor();
			float intensity = data.light->GetIntensity();
			float range = data.light->GetRange();

			constants.lightPosition[i] = Vector4(position.x, position.y, position.z, 1.0f);
			constants.lightColor[i] = Vector4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
			constants.lightAttenuation[i] = GetAttenuation(range);
			constants.lightDirection[i] = Vector4(0.0f, 0.0f, 1.0f, 0.0f);
		}

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraLightDataId, s_ConstantBuffer);
	}
}
