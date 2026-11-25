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
	GfxBuffer* PerCameraLightDataConstantBuffer::s_ProbeVolumesBuffer = nullptr;
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
		uint32_t flags;
		uint32_t shadowDataOffset;
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
		uint32_t flags;
		uint32_t shadowDataOffset;
		float dummy;
	};

	struct ShadowBufferData
	{
		Matrix worldToShadow;
		Vector4 shadowBounds;
	};

	struct ProbeVolumeBufferData
	{
		Vector3 positionMinWS;
		Vector3 positionMinVS;
		Vector3 positionMaxWS;
		Vector3 positionMaxVS;
		Vector3 invSizeWS;
		Vector3Uint topLevelSize;
		uint32_t probeAtlasOffset;
		float dummy;
	};

	struct ReflectionProbeBufferData
	{
		Vector3 positionWS;
		float squareRange;
		float volume;
		float fade;
		Vector3 positionMinWS;
		Vector3 positionMinVS;
		Vector3 positionMaxWS;
		Vector3 positionMaxVS;
		uint32_t index;
		uint32_t type;
	};

	static PointLightBufferData s_PointDatas[MAX_REALTIME_LIGHTS];
	static SpotLightBufferData s_SpotDatas[MAX_REALTIME_LIGHTS];
	static ProbeVolumeBufferData s_ProbeVolumeDatas[MAX_REALTIME_LIGHTS];
	static ReflectionProbeBufferData s_ReflectionProbeDatas[MAX_REALTIME_LIGHTS];
	static ShadowBufferData s_ShadowDatas[MAX_SHADOWS];

	static size_t s_PerCameraLightDataId = TO_HASH("PerCameraLightData");
	static size_t s_PointLightsDataId = TO_HASH("_PointLightsData");
	static size_t s_SpotLightsDataId = TO_HASH("_SpotLightsData");
	static size_t s_ShadowsDataId = TO_HASH("_ShadowsData");
	static size_t s_ProbeVolumesDataId = TO_HASH("_ProbeVolumesData");
	static size_t s_ReflectionProbesDataId = TO_HASH("_ReflectionProbesData");

	bool CompareReflectionProbes(const ReflectionProbeBufferData& d1, const ReflectionProbeBufferData& d2)
	{
		return d1.volume > d2.volume;
	}

	void PerCameraLightDataConstantBuffer::BindData(Camera* camera, Light* mainLight, SkyRenderer* skyRenderer, const List<Light*>& lights, const List<ProbeVolume*>& probeVolumes, const List<ReflectionProbe*>& reflectionProbes, const Vector2Int& shadowAtlasSize)
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

			BufferProperties probeVolumesBufferProperties = {};
			probeVolumesBufferProperties.elementCount = MAX_REALTIME_LIGHTS;
			probeVolumesBufferProperties.elementSize = sizeof(ProbeVolumeBufferData);
			probeVolumesBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;

			GfxDevice::CreateBuffer(probeVolumesBufferProperties, s_ProbeVolumesBuffer);

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
			Color color = light->GetColor().ToLinear();
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

		uint32_t pointOffset = 0;
		uint32_t spotOffset = 0;
		uint32_t reflectionProbeOffset = 0;
		uint32_t probeVolumeOffset = 0;
		uint32_t shadowOffset = 0;

		for (size_t i = 0; i < lights.size(); ++i)
		{
			Light* light = lights[i];

			Transform* transform = light->GetTransform();
			Vector3 positionWS = transform->GetPosition();
			Vector3 positionVS = Vector3::Transform(positionWS, view);
			bool hasShadow = light->IsCastingShadows();
			bool hasFog = light->IsCastingFog();
			Color color = light->GetColor().ToLinear();
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
						s_ShadowDatas[shadowOffset] = shadowData;
						++shadowOffset;
					}
				}

				s_PointDatas[pointOffset] = data;
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
					s_ShadowDatas[shadowOffset] = shadowData;
					++shadowOffset;
				}
				
				s_SpotDatas[spotOffset] = data;
				++spotOffset;
			}
		}
		for (size_t i = 0; i < probeVolumes.size(); ++i)
		{
			ProbeVolume* probeVolume = probeVolumes[i];
			if (probeVolume->m_Index != UINT32_MAX)
			{
				Vector3 positionWS = probeVolume->GetTransform()->GetPosition();
				Vector3 size = probeVolume->GetSize();
				Vector3Int volumeSize = probeVolume->GetDensity();
				
				Vector3 positionMinVS = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
				Vector3 positionMaxVS = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
				Vector3 halfSize = size * 0.5f;

				for (uint32_t j = 0; j < 8; ++j)
				{
					Vector3 cornerWS = positionWS + Vector3((j & 1 ? 1 : -1) * halfSize.x, (j & 2 ? 1 : -1) * halfSize.y, (j & 4 ? 1 : -1) * halfSize.z);
					Vector3 cornerVS = Vector3::Transform(cornerWS, view);
					positionMinVS = Vector3::Min(positionMinVS, cornerVS);
					positionMaxVS = Vector3::Max(positionMaxVS, cornerVS);
				}

				ProbeVolumeBufferData data;
				data.positionMinWS = positionWS - halfSize;
				data.positionMinVS = positionMinVS;
				data.positionMaxWS = positionWS + halfSize;
				data.positionMaxVS = positionMaxVS;
				data.invSizeWS = Vector3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z);
				data.topLevelSize = Vector3Uint(static_cast<uint32_t>(volumeSize.x), static_cast<uint32_t>(volumeSize.y), static_cast<uint32_t>(volumeSize.z));
				data.probeAtlasOffset = probeVolume->m_ProbeAtlasOffset;

				s_ProbeVolumeDatas[probeVolumeOffset] = data;
				++probeVolumeOffset;
			}
		}
		for (size_t i = 0; i < reflectionProbes.size(); ++i)
		{
			ReflectionProbe* reflectionProbe = reflectionProbes[i];
			uint32_t index = reflectionProbe->GetIndex();
			if (index != UINT32_MAX)
			{
				Transform* transform = reflectionProbe->GetTransform();
				Vector3 positionWS = transform->GetPosition();

				ReflectionProbeBufferData data;
				data.positionWS = positionWS;
				if (reflectionProbe->GetType() == ReflectionProbeType::Sphere)
				{
					float radius = std::max(reflectionProbe->GetRadius(), 1e-2f);
					data.positionMinWS = Vector3(radius, 0, 0);
					data.positionMinVS = Vector3::Transform(positionWS, view);
					data.squareRange = radius * radius;
					data.volume = ((4.0f * Math::Pi * radius * radius * radius) / 3.0f);
					data.type = 0;
				}
				else
				{
					Vector3 positionMinVS = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
					Vector3 positionMaxVS = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
					Vector3 size = reflectionProbe->GetSize();
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
					data.volume = (size.x * size.y * size.z);
					data.type = 1;
				}
				data.fade = std::max(reflectionProbe->GetFade(), 1e-2f);
				data.index = index;

				s_ReflectionProbeDatas[reflectionProbeOffset] = data;
				++reflectionProbeOffset;
			}
		}
		if (reflectionProbeOffset > 0)
		{
			std::sort(s_ReflectionProbeDatas, s_ReflectionProbeDatas + reflectionProbeOffset, CompareReflectionProbes);
		}
		constants.lightsCount = Vector4Uint(pointOffset, spotOffset, probeVolumeOffset, reflectionProbeOffset);

		float texelEpsilonX = 1.0f / shadowAtlasSize.x;
		float texelEpsilonY = 1.0f / shadowAtlasSize.y;
		constants.shadow3x3PCFTermC0 = Vector4(20.0f / 267.0f, 33.0f / 267.0f, 55.0f / 267.0f, 0.0f);
		constants.shadow3x3PCFTermC1 = Vector4(texelEpsilonX, texelEpsilonY, -texelEpsilonX, -texelEpsilonY);
		constants.shadow3x3PCFTermC2 = Vector4(texelEpsilonX, texelEpsilonY, 0.0f, 0.0f);
		constants.shadow3x3PCFTermC3 = Vector4(-texelEpsilonX, -texelEpsilonY, 0.0f, 0.0f);

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(PerCameraLightData));
		s_PointLightsBuffer->SetData(reinterpret_cast<char*>(&s_PointDatas), sizeof(PointLightBufferData) * pointOffset);
		s_SpotLightsBuffer->SetData(reinterpret_cast<char*>(&s_SpotDatas), sizeof(SpotLightBufferData) * spotOffset);
		s_ShadowsBuffer->SetData(reinterpret_cast<char*>(&s_ShadowDatas), sizeof(ShadowBufferData) * shadowOffset);
		s_ProbeVolumesBuffer->SetData(reinterpret_cast<char*>(&s_ProbeVolumeDatas), sizeof(ProbeVolumeBufferData)* probeVolumeOffset);
		s_ReflectionProbesBuffer->SetData(reinterpret_cast<char*>(&s_ReflectionProbeDatas), sizeof(ReflectionProbeBufferData)* reflectionProbeOffset);
		GfxDevice::SetGlobalBuffer(s_PerCameraLightDataId, s_ConstantBuffer);
		GfxDevice::SetGlobalBuffer(s_PointLightsDataId, s_PointLightsBuffer);
		GfxDevice::SetGlobalBuffer(s_SpotLightsDataId, s_SpotLightsBuffer);
		GfxDevice::SetGlobalBuffer(s_ShadowsDataId, s_ShadowsBuffer);
		GfxDevice::SetGlobalBuffer(s_ProbeVolumesDataId, s_ProbeVolumesBuffer);
		GfxDevice::SetGlobalBuffer(s_ReflectionProbesDataId, s_ReflectionProbesBuffer);
	}
}
