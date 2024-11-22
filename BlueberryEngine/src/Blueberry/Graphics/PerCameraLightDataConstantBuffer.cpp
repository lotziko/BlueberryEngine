#include "bbpch.h"
#include "PerCameraLightDataConstantBuffer.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"

#include "GfxDevice.h"
#include "GfxBuffer.h"

namespace Blueberry
{
	#define MAIN_LIGHT_CASCADES 3
	#define MAX_REALTIME_LIGHTS 128

	struct CONSTANTS
	{
		Vector4 mainLightColor;
		Vector4 mainLightDirection;
		Matrix mainWorldToShadow[MAIN_LIGHT_CASCADES + 1];
		Vector4 mainShadowBounds[MAIN_LIGHT_CASCADES + 1];
		Vector4 mainShadowCascades[MAIN_LIGHT_CASCADES];

		Vector4 lightsCount;
		Vector4 lightParam[MAX_REALTIME_LIGHTS];
		Vector4 lightPosition[MAX_REALTIME_LIGHTS];
		Vector4 lightColor[MAX_REALTIME_LIGHTS];
		Vector4 lightAttenuation[MAX_REALTIME_LIGHTS];
		Vector4 lightDirection[MAX_REALTIME_LIGHTS];
		Matrix worldToShadow[MAX_REALTIME_LIGHTS];
		Vector4 shadowBounds[MAX_REALTIME_LIGHTS];
		Vector4 shadow3x3PCFTermC0;
		Vector4 shadow3x3PCFTermC1;
		Vector4 shadow3x3PCFTermC2;
		Vector4 shadow3x3PCFTermC3;
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

	void PerCameraLightDataConstantBuffer::BindData(const LightData& mainLight, const std::vector<LightData>& lights)
	{
		static size_t perCameraLightDataId = TO_HASH("PerCameraLightData");

		if (s_ConstantBuffer == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		CONSTANTS constants = {};
		if (mainLight.light != nullptr)
		{
			Light* light = mainLight.light;
			constants.mainLightDirection = (Vector4)Vector3::Transform(Vector3::Backward, mainLight.transform->GetRotation());
			constants.mainLightColor = light->GetColor() * light->GetIntensity();
			for (int i = 0; i < light->m_SliceCount; ++i)
			{
				constants.mainWorldToShadow[i] = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[i]);
				constants.mainShadowBounds[i] = light->m_ShadowBounds[i];
				constants.mainShadowCascades[i] = light->m_ShadowCascades[i];
			}
		}
		uint8_t offset = 0;
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

			if (type == LightType::Spot)
			{
				Vector3 dir = Vector3::Transform(Vector3::Backward, data.transform->GetRotation());
				direction = Vector4(dir.x, dir.y, dir.z, 0.0f);
			}

			if (type == LightType::Directional)
			{
				Vector3 dir = Vector3::Transform(Vector3::Backward, data.transform->GetRotation());
				position = Vector4(dir.x, dir.y, dir.z, 0.0f);
			}
			else
			{
				Vector3 pos = data.transform->GetPosition();
				position = Vector4(pos.x, pos.y, pos.z, 1.0f);
			}

			Vector4 lightParam = Vector4(light->IsCastingShadows() ? 1 : 0, 0, 0, range * range);
			Vector4 lightPosition = position;
			Vector4 lightColor = Vector4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
			Vector4 lightAttenuation = GetAttenuation(type, range, outerSpotAngle, innerSpotAngle);
			Vector4 lightDirection = direction;

			for (int j = 0; j < light->m_SliceCount; ++j)
			{
				constants.lightParam[offset] = lightParam;
				constants.lightPosition[offset] = lightPosition;
				constants.lightColor[offset] = lightColor;
				constants.lightAttenuation[offset] = lightAttenuation;
				constants.lightDirection[offset] = lightDirection;
				constants.worldToShadow[offset] = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[j]);
				constants.shadowBounds[offset] = light->m_ShadowBounds[j];
				++offset;
			}
		}
		constants.lightsCount = Vector4(offset, 0.0f, 0.0f, 0.0f);

		float texelEpsilonX = 1.0f / 2048;
		float texelEpsilonY = 1.0f / 2048;
		constants.shadow3x3PCFTermC0 = Vector4(20.0f / 267.0f, 33.0f / 267.0f, 55.0f / 267.0f, 0.0f);
		constants.shadow3x3PCFTermC1 = Vector4(texelEpsilonX, texelEpsilonY, -texelEpsilonX, -texelEpsilonY);
		constants.shadow3x3PCFTermC2 = Vector4(texelEpsilonX, texelEpsilonY, 0.0f, 0.0f);
		constants.shadow3x3PCFTermC3 = Vector4(-texelEpsilonX, -texelEpsilonY, 0.0f, 0.0f);

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraLightDataId, s_ConstantBuffer);
	}
}
