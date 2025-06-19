#include "FogLightDataConstantBuffer.h"

#include "..\GfxDevice.h"
#include "..\GfxBuffer.h"
#include "..\LightHelper.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	#define MAIN_LIGHT_CASCADES 3
	#define MAX_REALTIME_LIGHTS 128

	struct FogLightData
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
		Matrix worldToCookie[MAX_REALTIME_LIGHTS];
		Vector4 shadow3x3PCFTermC0;
		Vector4 shadow3x3PCFTermC1;
		Vector4 shadow3x3PCFTermC2;
		Vector4 shadow3x3PCFTermC3;
	};

	static size_t s_FogLightDataId = TO_HASH("_FogLightData");

	void FogLightDataConstantBuffer::BindData(Light* mainLight, const List<Light*>& lights, const Vector2Int& shadowAtlasSize)
	{
		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(FogLightData) * 1;
			constantBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		}

		FogLightData constants = {};

		if (mainLight != nullptr)
		{
			Light* light = mainLight;
			constants.mainLightDirection = static_cast<Vector4>(Vector3::Transform(Vector3::Backward, mainLight->GetTransform()->GetRotation()));
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
			Light* light = lights[i];

			LightRenderingData renderingData;
			LightHelper::GetRenderingData(light, light->GetTransform(), renderingData);

			for (int j = 0; j < light->m_SliceCount; ++j)
			{
				constants.lightParam[offset] = renderingData.lightParam;
				constants.lightPosition[offset] = renderingData.lightPosition;
				constants.lightColor[offset] = renderingData.lightColor;
				constants.lightAttenuation[offset] = renderingData.lightAttenuation;
				constants.lightDirection[offset] = renderingData.lightDirection;
				constants.worldToShadow[offset] = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[j]);
				constants.shadowBounds[offset] = light->m_ShadowBounds[j];
				constants.worldToCookie[offset] = GfxDevice::GetGPUMatrix(light->m_WorldToCookie[j]);
				++offset;
			}
		}
		constants.lightsCount = Vector4(offset, 0.0f, 0.0f, 0.0f);

		float texelEpsilonX = 1.0f / shadowAtlasSize.x;
		float texelEpsilonY = 1.0f / shadowAtlasSize.y;
		constants.shadow3x3PCFTermC0 = Vector4(20.0f / 267.0f, 33.0f / 267.0f, 55.0f / 267.0f, 0.0f);
		constants.shadow3x3PCFTermC1 = Vector4(texelEpsilonX, texelEpsilonY, -texelEpsilonX, -texelEpsilonY);
		constants.shadow3x3PCFTermC2 = Vector4(texelEpsilonX, texelEpsilonY, 0.0f, 0.0f);
		constants.shadow3x3PCFTermC3 = Vector4(-texelEpsilonX, -texelEpsilonY, 0.0f, 0.0f);

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalBuffer(s_FogLightDataId, s_ConstantBuffer);
	}
}