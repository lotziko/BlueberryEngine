#include "PerCameraLightDataConstantBuffer.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "..\LightHelper.h"

namespace Blueberry
{
	#define MAIN_LIGHT_CASCADES 3
	#define MAX_REALTIME_LIGHTS 256

	struct PerCameraLightData
	{
		Vector3 mainLightColor;
		float mainLightHasShadow;
		Vector3 mainLightDirection;
		float mainLightHasFog;
		Matrix mainWorldToShadow[MAIN_LIGHT_CASCADES + 1];
		Vector4 mainShadowBounds[MAIN_LIGHT_CASCADES + 1];
		Vector4 mainShadowCascades[MAIN_LIGHT_CASCADES];
		Vector4 ambientLightColor;

		Vector4 lightsCount;
		Vector4 shadow3x3PCFTermC0;
		Vector4 shadow3x3PCFTermC1;
		Vector4 shadow3x3PCFTermC2;
		Vector4 shadow3x3PCFTermC3;
	};

	struct PointLightData
	{
		Vector3 positionWS;
		float hasShadow;
		Vector3 positionVS;
		float hasFog;
		Vector3 color;
		float squareRange;
		Vector4 attenuation;
		Matrix worldToShadow[6];
		Vector4 shadowBounds[6];
	};

	struct SpotLightData
	{
		Vector3 positionWS;
		float hasShadow;
		Vector3 positionVS;
		float hasFog;
		Vector3 color;
		float hasCookie;
		Vector4 attenuation;
		Vector3 directionWS;
		float range;
		Vector3 directionVS;
		float coneOuterAngle;
		Matrix worldToShadow;
		Vector4 shadowBounds;
		Matrix worldToCookie;
	};

	static size_t s_PerCameraLightDataId = TO_HASH("PerCameraLightData");
	static size_t s_PointLightsDataId = TO_HASH("_PointLightsData");
	static size_t s_SpotLightsDataId = TO_HASH("_SpotLightsData");

	void PerCameraLightDataConstantBuffer::BindData(Camera* camera, Light* mainLight, SkyRenderer* skyRenderer, const List<Light*>& lights, const Vector2Int& shadowAtlasSize)
	{
		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(PerCameraLightData) * 1;
			constantBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);

			BufferProperties pointLightsBufferProperties = {};
			pointLightsBufferProperties.type = BufferType::Structured;
			pointLightsBufferProperties.elementCount = MAX_REALTIME_LIGHTS;
			pointLightsBufferProperties.elementSize = sizeof(PointLightData);
			pointLightsBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(pointLightsBufferProperties, s_PointLightsBuffer);

			BufferProperties spotLightsBufferProperties = {};
			spotLightsBufferProperties.type = BufferType::Structured;
			spotLightsBufferProperties.elementCount = MAX_REALTIME_LIGHTS;
			spotLightsBufferProperties.elementSize = sizeof(SpotLightData);
			spotLightsBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(spotLightsBufferProperties, s_SpotLightsBuffer);
		}

		PerCameraLightData constants = {};
		Matrix view = camera->GetViewMatrix();

		if (mainLight != nullptr)
		{
			Light* light = mainLight;
			Color color = light->GetColor();
			float intensity = light->GetIntensity();
			Vector3 finalColor = Vector3(color.x * intensity, color.y * intensity, color.z * intensity);
			constants.mainLightDirection = Vector3::Transform(Vector3::Backward, mainLight->GetTransform()->GetRotation());
			constants.mainLightHasShadow = mainLight->IsCastingShadows();
			constants.mainLightColor = finalColor;
			constants.mainLightHasFog = mainLight->IsCastingFog();
			for (int i = 0; i < light->m_SliceCount; ++i)
			{
				constants.mainWorldToShadow[i] = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[i]);
				constants.mainShadowBounds[i] = light->m_ShadowBounds[i];
				constants.mainShadowCascades[i] = light->m_ShadowCascades[i];
			}
		}
		if (skyRenderer != nullptr)
		{
			constants.ambientLightColor = skyRenderer->GetAmbientColor();
		}

		PointLightData pointDatas[MAX_REALTIME_LIGHTS];
		SpotLightData spotDatas[MAX_REALTIME_LIGHTS];

		uint8_t pointOffset = 0;
		uint8_t spotOffset = 0;
		for (int i = 0; i < lights.size(); i++)
		{
			Light* light = lights[i];

			Transform* transform = light->GetTransform();
			Vector3 positionWS = transform->GetPosition();
			Vector3 positionVS = Vector3::Transform(positionWS, view);
			bool hasShadow = light->IsCastingShadows();
			bool hasFog = light->IsCastingFog();
			Color color = light->GetColor();
			float intensity = light->GetIntensity();
			Vector3 finalColor = Vector3(color.x * intensity, color.y * intensity, color.z * intensity);
			float range = light->GetRange();
			LightType lightType = light->GetType();

			if (lightType == LightType::Point)
			{
				PointLightData data;
				data.positionWS = positionWS;
				data.hasShadow = hasShadow;
				data.positionVS = positionVS;
				data.hasFog = hasFog;
				data.color = finalColor;
				data.squareRange = range * range;
				data.attenuation = LightHelper::GetAttenuation(LightType::Point, light->GetRange(), 0, 0);
				for (int i = 0; i < 6; ++i)
				{
					data.worldToShadow[i] = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[i]);
					data.shadowBounds[i] = light->m_ShadowBounds[i];
				}
				pointDatas[pointOffset] = data;
				++pointOffset;
			}
			else if (lightType == LightType::Spot)
			{
				SpotLightData data;
				data.positionWS = positionWS;
				data.hasShadow = hasShadow;
				data.positionVS = positionVS;
				data.hasFog = hasFog;
				data.color = finalColor;
				data.hasCookie = light->GetCookie() != nullptr;
				data.attenuation = LightHelper::GetAttenuation(LightType::Spot, light->GetRange(), light->GetOuterSpotAngle(), light->GetInnerSpotAngle());
				data.directionWS = Vector3::Transform(Vector3::Backward, transform->GetRotation());
				data.range = light->GetRange();
				data.directionVS = static_cast<Vector3>(Vector4::Transform(Vector4(data.directionWS.x, data.directionWS.y, data.directionWS.z, 0.0f), view));
				data.coneOuterAngle = ToRadians(light->GetOuterSpotAngle());
				data.worldToShadow = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[0]);
				data.shadowBounds = light->m_ShadowBounds[0];
				data.worldToCookie = GfxDevice::GetGPUMatrix(light->m_WorldToCookie[0]);
				spotDatas[spotOffset] = data;
				++spotOffset;
			}
		}
		constants.lightsCount = Vector4(pointOffset, spotOffset, 0.0f, 0.0f);

		float texelEpsilonX = 1.0f / shadowAtlasSize.x;
		float texelEpsilonY = 1.0f / shadowAtlasSize.y;
		constants.shadow3x3PCFTermC0 = Vector4(20.0f / 267.0f, 33.0f / 267.0f, 55.0f / 267.0f, 0.0f);
		constants.shadow3x3PCFTermC1 = Vector4(texelEpsilonX, texelEpsilonY, -texelEpsilonX, -texelEpsilonY);
		constants.shadow3x3PCFTermC2 = Vector4(texelEpsilonX, texelEpsilonY, 0.0f, 0.0f);
		constants.shadow3x3PCFTermC3 = Vector4(-texelEpsilonX, -texelEpsilonY, 0.0f, 0.0f);

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		s_PointLightsBuffer->SetData(reinterpret_cast<char*>(&pointDatas), sizeof(PointLightData) * pointOffset);
		s_SpotLightsBuffer->SetData(reinterpret_cast<char*>(&spotDatas), sizeof(SpotLightData) * spotOffset);
		GfxDevice::SetGlobalBuffer(s_PerCameraLightDataId, s_ConstantBuffer);
		GfxDevice::SetGlobalBuffer(s_PointLightsDataId, s_PointLightsBuffer);
		GfxDevice::SetGlobalBuffer(s_SpotLightsDataId, s_SpotLightsBuffer);
	}
}
