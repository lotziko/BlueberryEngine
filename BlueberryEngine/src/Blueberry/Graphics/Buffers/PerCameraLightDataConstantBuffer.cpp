#include "PerCameraLightDataConstantBuffer.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "..\LightHelper.h"

namespace Blueberry
{
	GfxBuffer* PerCameraLightDataConstantBuffer::s_ConstantBuffer = nullptr;
	GfxBuffer* PerCameraLightDataConstantBuffer::s_PointLightsBuffer = nullptr;
	GfxBuffer* PerCameraLightDataConstantBuffer::s_SpotLightsBuffer = nullptr;
	GfxBuffer* PerCameraLightDataConstantBuffer::s_ShadowsBuffer = nullptr;
	GfxBuffer* PerCameraLightDataConstantBuffer::s_ReflectionProbesBuffer = nullptr;

	#define MAIN_LIGHT_CASCADES 3
	#define MAX_REALTIME_LIGHTS 256
	#define MAX_SHADOWS 1024

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

		Vector4Uint lightsCount;
		Vector4 probeVolumeMin;
		Vector4 probeVolumeSize;
		Vector4 probeVolumeInvSize;
		Vector4 probeVolumeCellSize;
		Vector4 shadow3x3PCFTermC0;
		Vector4 shadow3x3PCFTermC1;
		Vector4 shadow3x3PCFTermC2;
		Vector4 shadow3x3PCFTermC3;
	};

	struct PointLightBufferData
	{
		Vector3 positionWS;
		Vector3 positionVS;
		Vector3 color;
		float squareRange;
		Vector4 attenuation;
		unsigned int flags;
		unsigned int shadowDataOffset;
	};

	struct SpotLightBufferData
	{
		Vector3 positionWS;
		Vector3 positionVS;
		Vector3 color;
		Vector4 attenuation;
		Vector3 directionWS;
		float range;
		Vector3 directionVS;
		float coneOuterAngle;
		Matrix worldToCookie;
		unsigned int flags;
		unsigned int shadowDataOffset;
		float dummy;
	};

	struct ShadowBufferData
	{
		Matrix worldToShadow;
		Vector4 shadowBounds;
	};

	struct ReflectionProbeBufferData
	{
		Vector3 positionWS;
		float squareRange;
		float weight;
		float fade;
		Vector3 positionMinWS;
		Vector3 positionMinVS;
		Vector3 positionMaxWS;
		Vector3 positionMaxVS;
		unsigned int index;
		unsigned int type;
	};

	static size_t s_PerCameraLightDataId = TO_HASH("PerCameraLightData");
	static size_t s_PointLightsDataId = TO_HASH("_PointLightsData");
	static size_t s_SpotLightsDataId = TO_HASH("_SpotLightsData");
	static size_t s_ShadowsDataId = TO_HASH("_ShadowsData");
	static size_t s_ReflectionProbesDataId = TO_HASH("_ReflectionProbesData");

	void PerCameraLightDataConstantBuffer::BindData(Camera* camera, Light* mainLight, SkyRenderer* skyRenderer, ProbeVolume* probeVolume, const List<Light*>& lights, const List<ReflectionProbe*>& reflectionProbes, const Vector2Int& shadowAtlasSize)
	{
		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(PerCameraLightData) * 1;
			constantBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);

			BufferProperties pointLightsBufferProperties = {};
			pointLightsBufferProperties.elementCount = MAX_REALTIME_LIGHTS;
			pointLightsBufferProperties.elementSize = sizeof(PointLightBufferData);
			pointLightsBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;

			GfxDevice::CreateBuffer(pointLightsBufferProperties, s_PointLightsBuffer);

			BufferProperties spotLightsBufferProperties = {};
			spotLightsBufferProperties.elementCount = MAX_REALTIME_LIGHTS;
			spotLightsBufferProperties.elementSize = sizeof(SpotLightBufferData);
			spotLightsBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;

			GfxDevice::CreateBuffer(spotLightsBufferProperties, s_SpotLightsBuffer);

			BufferProperties shadowsBufferProperties = {};
			shadowsBufferProperties.elementCount = MAX_SHADOWS;
			shadowsBufferProperties.elementSize = sizeof(ShadowBufferData);
			shadowsBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;

			GfxDevice::CreateBuffer(shadowsBufferProperties, s_ShadowsBuffer);

			BufferProperties reflectionProbesBufferProperties = {};
			reflectionProbesBufferProperties.elementCount = MAX_REALTIME_LIGHTS;
			reflectionProbesBufferProperties.elementSize = sizeof(ReflectionProbeBufferData);
			reflectionProbesBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;

			GfxDevice::CreateBuffer(reflectionProbesBufferProperties, s_ReflectionProbesBuffer);
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
			for (uint8_t i = 0; i < LightHelper::GetSliceCount(light->GetType()); ++i)
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
		else
		{
			constants.ambientLightColor = Vector4::One;
		}

		PointLightBufferData pointDatas[MAX_REALTIME_LIGHTS];
		SpotLightBufferData spotDatas[MAX_REALTIME_LIGHTS];
		ReflectionProbeBufferData reflectionProbeDatas[MAX_REALTIME_LIGHTS];
		ShadowBufferData shadowDatas[MAX_SHADOWS];

		uint32_t pointOffset = 0;
		uint32_t spotOffset = 0;
		uint32_t reflectionProbeOffset = 0;
		uint32_t shadowOffset = 0;

		for (size_t i = 0; i < lights.size(); ++i)
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
				float squareRange = range * range;
				Vector4 attenuation = LightHelper::GetAttenuation(LightType::Point, light->GetRange(), 0, 0);

				PointLightBufferData data;
				data.positionWS = positionWS;
				data.positionVS = positionVS;
				data.color = finalColor;
				data.squareRange = squareRange;
				data.attenuation = attenuation;
				data.flags = (hasShadow ? 1 : 0) | (hasFog ? 2 : 0);

				if (hasShadow)
				{
					data.shadowDataOffset = shadowOffset;

					for (int i = 0; i < 6; ++i)
					{
						ShadowBufferData shadowData;
						shadowData.worldToShadow = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[i]);
						shadowData.shadowBounds = light->m_ShadowBounds[i];
						shadowDatas[shadowOffset] = shadowData;
						++shadowOffset;
					}
				}

				pointDatas[pointOffset] = data;
				++pointOffset;
			}
			else if (lightType == LightType::Spot)
			{
				bool hasCookie = light->GetCookie() != nullptr;

				SpotLightBufferData data;
				data.positionWS = positionWS;
				data.positionVS = positionVS;
				data.color = finalColor;
				data.attenuation = LightHelper::GetAttenuation(LightType::Spot, light->GetRange(), light->GetOuterSpotAngle(), light->GetInnerSpotAngle());
				data.directionWS = Vector3::Transform(Vector3::Backward, transform->GetRotation());
				data.range = light->GetRange();
				data.directionVS = static_cast<Vector3>(Vector4::Transform(Vector4(data.directionWS.x, data.directionWS.y, data.directionWS.z, 0.0f), view));
				data.coneOuterAngle = Math::ToRadians(light->GetOuterSpotAngle());
				data.worldToCookie = hasCookie ? GfxDevice::GetGPUMatrix(light->m_WorldToCookie) : Matrix::Identity;
				data.flags = (hasShadow ? 1 : 0) | (hasFog ? 2 : 0) | (hasCookie ? 4 : 0);
				
				if (hasShadow)
				{
					data.shadowDataOffset = shadowOffset;

					ShadowBufferData shadowData;
					shadowData.worldToShadow = GfxDevice::GetGPUMatrix(light->m_AtlasWorldToShadow[0]);
					shadowData.shadowBounds = light->m_ShadowBounds[0];
					shadowDatas[shadowOffset] = shadowData;
					++shadowOffset;
				}
				
				spotDatas[spotOffset] = data;
				++spotOffset;
			}
		}
		for (size_t i = 0; i < reflectionProbes.size(); ++i)
		{
			ReflectionProbe* reflectionProbe = reflectionProbes[i];
			if (reflectionProbe->m_AtlasIndex != UINT_MAX)
			{
				Transform* transform = reflectionProbe->GetTransform();
				Vector3 positionWS = transform->GetPosition();

				ReflectionProbeBufferData data;
				data.positionWS = positionWS;
				if (reflectionProbe->GetType() == ReflectionProbeType::Sphere)
				{
					float radius = std::max(reflectionProbe->m_Radius, 1e-2f);
					data.positionMinWS = Vector3(radius, 0, 0);
					data.positionMinVS = Vector3::Transform(positionWS, view);
					data.squareRange = radius * radius;
					data.weight = 1.0f / ((4.0f * Math::Pi * radius * radius * radius) / 3.0f);
					data.type = 0;
				}
				else
				{
					Vector3 positionMinVS = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
					Vector3 positionMaxVS = Vector3(FLT_MIN, FLT_MIN, FLT_MIN);
					Vector3 size = reflectionProbe->m_Size;
					Vector3 halfSize = size * 0.5f;

					for (uint32_t j = 0; j < 8; ++j)
					{
						Vector3 cornerWS = positionWS + Vector3((j & 1 ? 1 : -1) * halfSize.x, (j & 2 ? 1 : -1) * halfSize.y, (j & 4 ? 1 : -1) * halfSize.z);
						Vector3 cornerVS = Vector3::Transform(cornerWS, view);
						positionMinVS = Vector3::Min(positionMinVS, cornerVS);
						positionMaxVS = Vector3::Max(positionMaxVS, cornerVS);
					}

					data.positionMinWS = positionWS - halfSize;
					data.positionMinVS = positionMinVS;
					data.positionMaxWS = positionWS + halfSize;
					data.positionMaxVS = positionMaxVS;
					data.weight = 1.0f / (size.x * size.y * size.z);
					data.type = 1;
				}
				data.fade = std::max(reflectionProbe->m_Fade, 1e-2f);
				data.index = reflectionProbe->m_AtlasIndex;
				
				reflectionProbeDatas[reflectionProbeOffset] = data;
				++reflectionProbeOffset;
			}
		}
		constants.lightsCount = Vector4Uint(pointOffset, spotOffset, reflectionProbeOffset, 0u);

		if (probeVolume != nullptr)
		{
			AABB bounds = probeVolume->GetBounds();
			Vector3 center = bounds.Center;
			Vector3 extents = bounds.Extents;
			Vector3 size = extents * 2;
			Vector3Int intSize = probeVolume->GetSize();
			constants.probeVolumeMin = Vector4(center - extents);
			constants.probeVolumeSize = Vector4(static_cast<float>(intSize.x), static_cast<float>(intSize.y), static_cast<float>(intSize.z), 0.0f);
			constants.probeVolumeInvSize = Vector4(1.0f / (extents.x * 2), 1.0f / (extents.y * 2), 1.0f / (extents.z * 2), 0);
			constants.probeVolumeCellSize = Vector4(size.x / intSize.x, size.y / intSize.y, size.z / intSize.z, 0);
		}

		float texelEpsilonX = 1.0f / shadowAtlasSize.x;
		float texelEpsilonY = 1.0f / shadowAtlasSize.y;
		constants.shadow3x3PCFTermC0 = Vector4(20.0f / 267.0f, 33.0f / 267.0f, 55.0f / 267.0f, 0.0f);
		constants.shadow3x3PCFTermC1 = Vector4(texelEpsilonX, texelEpsilonY, -texelEpsilonX, -texelEpsilonY);
		constants.shadow3x3PCFTermC2 = Vector4(texelEpsilonX, texelEpsilonY, 0.0f, 0.0f);
		constants.shadow3x3PCFTermC3 = Vector4(-texelEpsilonX, -texelEpsilonY, 0.0f, 0.0f);

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		s_PointLightsBuffer->SetData(reinterpret_cast<char*>(&pointDatas), sizeof(PointLightBufferData) * pointOffset);
		s_SpotLightsBuffer->SetData(reinterpret_cast<char*>(&spotDatas), sizeof(SpotLightBufferData) * spotOffset);
		s_ShadowsBuffer->SetData(reinterpret_cast<char*>(&shadowDatas), sizeof(ShadowBufferData) * shadowOffset);
		s_ReflectionProbesBuffer->SetData(reinterpret_cast<char*>(&reflectionProbeDatas), sizeof(ReflectionProbeBufferData) * reflectionProbeOffset);
		GfxDevice::SetGlobalBuffer(s_PerCameraLightDataId, s_ConstantBuffer);
		GfxDevice::SetGlobalBuffer(s_PointLightsDataId, s_PointLightsBuffer);
		GfxDevice::SetGlobalBuffer(s_SpotLightsDataId, s_SpotLightsBuffer);
		GfxDevice::SetGlobalBuffer(s_ShadowsDataId, s_ShadowsBuffer);
		GfxDevice::SetGlobalBuffer(s_ReflectionProbesDataId, s_ReflectionProbesBuffer);
	}
}
