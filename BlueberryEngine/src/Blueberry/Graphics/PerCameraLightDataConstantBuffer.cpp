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

	Vector4 GetAttenuation(LightType type, float lightRange, float spotOuterAngle, float spotInnerAngle)
	{
		Vector4 lightAttenuation = Vector4(0.0f, 1.0f, 0.0f, 1.0f);

		if (type != LightType::Directional)
		{
			float lightRangeSqr = lightRange * lightRange;
			float fadeStartDistanceSqr = 0.8f * 0.8f * lightRangeSqr;
			float fadeRangeSqr = (fadeStartDistanceSqr - lightRangeSqr);
			float lightRangeSqrOverFadeRangeSqr = -lightRangeSqr / fadeRangeSqr;
			float oneOverLightRangeSqr = 1.0f / Max(0.0001f, lightRangeSqr);

			lightAttenuation.x = oneOverLightRangeSqr;
			lightAttenuation.y = lightRangeSqrOverFadeRangeSqr;
		}

		if (type == LightType::Spot)
		{
			float cosOuterAngle = cos(ToRadians(spotOuterAngle) * 0.5f);
			float cosInnerAngle = cos(ToRadians(spotInnerAngle) * 0.5f);
			float smoothAngleRange = Max(0.001f, cosInnerAngle - cosOuterAngle);
			float invAngleRange = 1.0f / smoothAngleRange;
			float add = -cosOuterAngle * invAngleRange;

			lightAttenuation.z = invAngleRange;
			lightAttenuation.w = add;
		}
		
		return lightAttenuation;
	}

	void PerCameraLightDataConstantBuffer::BindData(const std::vector<LightData> lights)
	{
		static size_t perCameraLightDataId = TO_HASH("PerCameraLightData");

		if (s_ConstantBuffer == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		CONSTANTS constants;
		constants.lightsCount = Vector4((float)lights.size(), 0.0f, 0.0f, 0.0f);
		for (int i = 0; i < lights.size(); i++)
		{
			LightData data = lights[i];
			Light* light = data.light;
			Vector4 position;
			Vector4 direction = Vector4(0.0f, 0.0f, 1.0f, 0.0f);
			LightType type = light->GetType();
			Color color = light->GetColor();
			float intensity = light->GetIntensity();
			float range = light->GetRange();
			float outerSpotAngle = light->GetOuterSpotAngle();
			float innerSpotAngle = light->GetInnerSpotAngle();

			// Remember that Vector3::Forward is reversed
			if (type == LightType::Spot)
			{
				Vector3 dir = Vector3::Transform(Vector3::Forward, data.transform->GetRotation());
				direction = Vector4(dir.x, dir.y, dir.z, 0.0f);
			}

			if (type == LightType::Directional)
			{
				Vector3 dir = Vector3::Transform(Vector3::Forward, data.transform->GetRotation());
				position = Vector4(dir.x, dir.y, dir.z, 0.0f);
			}
			else
			{
				Vector3 pos = data.transform->GetPosition();
				position = Vector4(pos.x, pos.y, pos.z, 1.0f);
			}

			constants.lightPosition[i] = position;
			constants.lightColor[i] = Vector4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
			constants.lightAttenuation[i] = GetAttenuation(type, range, outerSpotAngle, innerSpotAngle);
			constants.lightDirection[i] = direction;
		}

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraLightDataId, s_ConstantBuffer);
	}
}
