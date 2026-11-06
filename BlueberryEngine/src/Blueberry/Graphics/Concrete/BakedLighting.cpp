#include "BakedLighting.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\LightingSettings.h"
#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\TextureCubeArray.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\GfxTexturePool.h"

namespace Blueberry
{
	ComputeShader* BakedLighting::s_FillProbeAtlasShader = nullptr;
	GfxBuffer* BakedLighting::s_FillProbeAtlasData = nullptr;
	GfxBuffer* BakedLighting::s_BricksData = nullptr;
	GfxBuffer* BakedLighting::s_ColorsData = nullptr;

	struct PerLightingSettingsData
	{
		GfxBuffer* scaleOffsetBuffer = nullptr;
		GfxTexture* probeVolumeTexture0 = nullptr;
		GfxTexture* probeVolumeTexture1 = nullptr;
		GfxTexture* probeVolumeTexture2 = nullptr;
		GfxTexture* probeVolumeTexture3 = nullptr;
		GfxTexture* probeVolumeTexture4 = nullptr;
		GfxTexture* probeVolumeTexture5 = nullptr;
		GfxTexture* reflectionProbeTexture = nullptr;

		uint32_t lightmapUpdateCount = UINT32_MAX;
		uint32_t probeVolumesUpdateCount = UINT32_MAX;
		uint32_t reflectionProbesUpdateCount = UINT32_MAX;
	};

	struct FillProbeAtlasData
	{
		uint32_t brickGroupsX;
		uint32_t firstBrickOffset;
		uint32_t brickCount;
		uint32_t colorSize;
	};

	static Dictionary<uint64_t, PerLightingSettingsData> s_PerLightingSettingsData = {};
	static float s_ReleaseResourcesTimer = 0.0f;

	#define MAX_BRICK_COUNT 8192

	static size_t s_LightmapTextureId = TO_HASH("_LightmapTexture");
	static size_t s_ProbeVolumeTexture0Id = TO_HASH("_ProbeVolumeTexture0");
	static size_t s_ProbeVolumeTexture1Id = TO_HASH("_ProbeVolumeTexture1");
	static size_t s_ProbeVolumeTexture2Id = TO_HASH("_ProbeVolumeTexture2");
	static size_t s_ProbeVolumeTexture3Id = TO_HASH("_ProbeVolumeTexture3");
	static size_t s_ProbeVolumeTexture4Id = TO_HASH("_ProbeVolumeTexture4");
	static size_t s_ProbeVolumeTexture5Id = TO_HASH("_ProbeVolumeTexture5");
	static size_t s_ReflectionTextureId = TO_HASH("_ReflectionTexture");
	static size_t s_PerLightmapInstanceDataId = TO_HASH("_PerLightmapInstanceData");
	static size_t s_FillProbeAtlasDataId = TO_HASH("FillProbeAtlasData");
	static size_t s_BricksDataId = TO_HASH("_BricksData");
	static size_t s_ColorsDataId = TO_HASH("_ColorsData");
	static size_t s_ProbeVolumeOutputTexture0Id = TO_HASH("_ProbeVolumeOutputTexture0");
	static size_t s_ProbeVolumeOutputTexture1Id = TO_HASH("_ProbeVolumeOutputTexture1");
	static size_t s_ProbeVolumeOutputTexture2Id = TO_HASH("_ProbeVolumeOutputTexture2");
	static size_t s_ProbeVolumeOutputTexture3Id = TO_HASH("_ProbeVolumeOutputTexture3");
	static size_t s_ProbeVolumeOutputTexture4Id = TO_HASH("_ProbeVolumeOutputTexture4");
	static size_t s_ProbeVolumeOutputTexture5Id = TO_HASH("_ProbeVolumeOutputTexture5");
	
	void BakedLighting::Initialize()
	{
		s_FillProbeAtlasShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/FillProbeAtlas.compute"));

		BufferProperties fillProbeAtlasBufferProperties = {};
		fillProbeAtlasBufferProperties.elementCount = 1;
		fillProbeAtlasBufferProperties.elementSize = sizeof(FillProbeAtlasData) * 1;
		fillProbeAtlasBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(fillProbeAtlasBufferProperties, s_FillProbeAtlasData);

		BufferProperties bricksBufferProperties = {};
		bricksBufferProperties.elementCount = MAX_BRICK_COUNT;
		bricksBufferProperties.elementSize = sizeof(ProbeVolumeBrickData);
		bricksBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;

		GfxDevice::CreateBuffer(bricksBufferProperties, s_BricksData);

		BufferProperties colorsBufferProperties = {};
		colorsBufferProperties.elementCount = MAX_BRICK_COUNT * 64 * 6;
		colorsBufferProperties.elementSize = sizeof(uint32_t);
		colorsBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;

		GfxDevice::CreateBuffer(colorsBufferProperties, s_ColorsData);
	}

	void BakedLighting::Shutdown()
	{
		Object::Destroy(s_FillProbeAtlasShader);
	}

	void BakedLighting::Apply(Scene* scene)
	{
		ReleaseIfNeeded();
		LightingSettings* lightingSettings = scene->GetLightingSettings();

		if (lightingSettings == nullptr)
		{
			GfxDevice::SetGlobalTexture(s_LightmapTextureId, DefaultTextures::GetWhite2D()->Get());
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture0Id, DefaultTextures::GetBlack3D()->Get());
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture1Id, DefaultTextures::GetBlack3D()->Get());
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture2Id, DefaultTextures::GetBlack3D()->Get());
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture3Id, DefaultTextures::GetBlack3D()->Get());
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture4Id, DefaultTextures::GetBlack3D()->Get());
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture5Id, DefaultTextures::GetBlack3D()->Get());
			GfxDevice::SetGlobalTexture(s_ReflectionTextureId, DefaultTextures::GetBlackCubeArray()->Get());
		}
		else
		{
			uint64_t key = static_cast<uint64_t>(lightingSettings->GetObjectId()) << 32 | static_cast<uint64_t>(scene->GetIndex());
			PerLightingSettingsData& perLightingSettingsData = s_PerLightingSettingsData[key];	// TODO clear every n sec if lightingSettings is invalid
			
			uint32_t lightmapUpdateCount = lightingSettings->GetLightmapUpdateCount();
			Texture2D* lightmap = lightingSettings->GetLightmap();
			if (perLightingSettingsData.lightmapUpdateCount != lightmapUpdateCount)
			{
				auto& chartOffsetScale = lightingSettings->GetChartOffsetScale();
				if (perLightingSettingsData.scaleOffsetBuffer != nullptr && perLightingSettingsData.scaleOffsetBuffer->GetElementCount() < chartOffsetScale.size())
				{
					delete perLightingSettingsData.scaleOffsetBuffer;
					perLightingSettingsData.scaleOffsetBuffer = nullptr;
				}
				if (perLightingSettingsData.scaleOffsetBuffer == nullptr)
				{
					BufferProperties properties = {};
					properties.elementCount = static_cast<uint32_t>(chartOffsetScale.size());
					properties.elementSize = sizeof(Vector4);
					properties.data = chartOffsetScale.data();
					properties.dataSize = chartOffsetScale.size() * sizeof(Vector4);
					properties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource;
					GfxDevice::CreateBuffer(properties, perLightingSettingsData.scaleOffsetBuffer);
				}
				else
				{
					perLightingSettingsData.scaleOffsetBuffer->SetData(chartOffsetScale.data(), chartOffsetScale.size() * sizeof(Vector4));
				}
				perLightingSettingsData.lightmapUpdateCount = lightmapUpdateCount;
			}
			GfxDevice::SetGlobalTexture(s_LightmapTextureId, lightmap == nullptr ? DefaultTextures::GetWhite2D()->Get() : lightmap->Get());
			GfxDevice::SetGlobalBuffer(s_PerLightmapInstanceDataId, perLightingSettingsData.scaleOffsetBuffer);

			uint32_t probeVolumesUpdateCount = lightingSettings->GetProbeVolumesUpdateCount();
			if (perLightingSettingsData.probeVolumesUpdateCount != probeVolumesUpdateCount)
			{
				auto& probeVolumes = lightingSettings->GetProbeVolumes();
				if (perLightingSettingsData.probeVolumeTexture0 == nullptr)
				{
					TextureProperties textureProperties = {};
					textureProperties.width = 512;
					textureProperties.height = 512;
					textureProperties.depth = 40;
					textureProperties.antiAliasing = 1;
					textureProperties.mipCount = 1;
					textureProperties.format = TextureFormat::R11G11B10_Float;
					textureProperties.dimension = TextureDimension::Texture3D;
					textureProperties.wrapMode = WrapMode::Clamp;
					textureProperties.filterMode = FilterMode::Trilinear;
					textureProperties.usageFlags = TextureUsageFlags::UnorderedAccess;

					perLightingSettingsData.probeVolumeTexture0 = GfxTexturePool::Get(textureProperties);
					perLightingSettingsData.probeVolumeTexture1 = GfxTexturePool::Get(textureProperties);
					perLightingSettingsData.probeVolumeTexture2 = GfxTexturePool::Get(textureProperties);
					perLightingSettingsData.probeVolumeTexture3 = GfxTexturePool::Get(textureProperties);
					perLightingSettingsData.probeVolumeTexture4 = GfxTexturePool::Get(textureProperties);
					perLightingSettingsData.probeVolumeTexture5 = GfxTexturePool::Get(textureProperties);
				}
				uint32_t brickOffset = 0;
				for (size_t i = 0; i < probeVolumes.size(); ++i)
				{
					ProbeVolumeInfo* info = probeVolumes[i].Get();
					if (info != nullptr)
					{
						auto& bricks = info->GetBricks();
						auto& colors = info->GetColors();
						if (bricks.size() > MAX_BRICK_COUNT)
						{
							BB_ERROR("Too many bricks in volume.");
							continue;
						}
						uint32_t brickCount = static_cast<uint32_t>(bricks.size());
						if (brickCount > 0)
						{
							uint32_t groupsX = std::min(brickCount, 65535u);
							uint32_t groupsY = (brickCount + groupsX - 1) / groupsX;

							FillProbeAtlasData fillProbeAtlasConstants = {};
							fillProbeAtlasConstants.brickGroupsX = groupsX;
							fillProbeAtlasConstants.firstBrickOffset = brickOffset;
							fillProbeAtlasConstants.brickCount = brickCount;
							fillProbeAtlasConstants.colorSize = static_cast<uint32_t>(info->GetColors().size() / (6 * sizeof(uint32_t)));
							s_FillProbeAtlasData->SetData(&fillProbeAtlasConstants, sizeof(FillProbeAtlasData));

							s_BricksData->SetData(bricks.data(), bricks.size() * sizeof(ProbeVolumeBrickData));
							s_ColorsData->SetData(colors.data(), colors.size());

							GfxDevice::SetGlobalBuffer(s_FillProbeAtlasDataId, s_FillProbeAtlasData);
							GfxDevice::SetGlobalBuffer(s_BricksDataId, s_BricksData);
							GfxDevice::SetGlobalBuffer(s_ColorsDataId, s_ColorsData);
							GfxDevice::SetGlobalTexture(s_ProbeVolumeOutputTexture0Id, perLightingSettingsData.probeVolumeTexture0);
							GfxDevice::SetGlobalTexture(s_ProbeVolumeOutputTexture1Id, perLightingSettingsData.probeVolumeTexture1);
							GfxDevice::SetGlobalTexture(s_ProbeVolumeOutputTexture2Id, perLightingSettingsData.probeVolumeTexture2);
							GfxDevice::SetGlobalTexture(s_ProbeVolumeOutputTexture3Id, perLightingSettingsData.probeVolumeTexture3);
							GfxDevice::SetGlobalTexture(s_ProbeVolumeOutputTexture4Id, perLightingSettingsData.probeVolumeTexture4);
							GfxDevice::SetGlobalTexture(s_ProbeVolumeOutputTexture5Id, perLightingSettingsData.probeVolumeTexture5);
							GfxDevice::Dispatch(s_FillProbeAtlasShader, 0, groupsX, groupsY, 1);
						}
						for (ProbeVolume* probeVolume : scene->GetComponents<ProbeVolume>())
						{
							if (probeVolume->GetIndex() == i)
							{
								probeVolume->m_ProbeAtlasOffset = brickOffset;
							}
						}
						brickOffset += brickCount;
					}
				}
				perLightingSettingsData.probeVolumesUpdateCount = probeVolumesUpdateCount;
			}
			if (perLightingSettingsData.probeVolumeTexture0 == nullptr)
			{
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture0Id, DefaultTextures::GetBlack3D()->Get());
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture1Id, DefaultTextures::GetBlack3D()->Get());
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture2Id, DefaultTextures::GetBlack3D()->Get());
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture3Id, DefaultTextures::GetBlack3D()->Get());
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture4Id, DefaultTextures::GetBlack3D()->Get());
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture5Id, DefaultTextures::GetBlack3D()->Get());
			}
			else
			{
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture0Id, perLightingSettingsData.probeVolumeTexture0);
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture1Id, perLightingSettingsData.probeVolumeTexture1);
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture2Id, perLightingSettingsData.probeVolumeTexture2);
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture3Id, perLightingSettingsData.probeVolumeTexture3);
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture4Id, perLightingSettingsData.probeVolumeTexture4);
				GfxDevice::SetGlobalTexture(s_ProbeVolumeTexture5Id, perLightingSettingsData.probeVolumeTexture5);
			}

			uint32_t reflectionProbesUpdateCount = lightingSettings->GetReflectionProbesUpdateCount();
			if (perLightingSettingsData.reflectionProbesUpdateCount != reflectionProbesUpdateCount)
			{
				auto& reflectionProbes = lightingSettings->GetReflectionProbes();
				if (perLightingSettingsData.reflectionProbeTexture != nullptr && perLightingSettingsData.reflectionProbeTexture->GetDepth() < reflectionProbes.size())
				{
					GfxTexturePool::Release(perLightingSettingsData.reflectionProbeTexture);
					perLightingSettingsData.reflectionProbeTexture = nullptr;
				}
				if (perLightingSettingsData.reflectionProbeTexture == nullptr)
				{
					TextureProperties textureProperties = {};
					textureProperties.width = 128;
					textureProperties.height = 128;
					textureProperties.depth = static_cast<uint32_t>(reflectionProbes.size());
					textureProperties.antiAliasing = 1;
					textureProperties.mipCount = 6;
					textureProperties.format = TextureFormat::BC6H_UFloat;
					textureProperties.dimension = TextureDimension::TextureCubeArray;
					textureProperties.wrapMode = WrapMode::Clamp;
					textureProperties.filterMode = FilterMode::Bilinear;

					perLightingSettingsData.reflectionProbeTexture = GfxTexturePool::Get(textureProperties);
				}
				// TODO use SetData and do not load probe data to GPU
				for (size_t i = 0; i < reflectionProbes.size(); ++i)
				{
					GfxTexture* sourceTexture = reflectionProbes[i]->Get();
					uint32_t mipLevels = sourceTexture->GetMipLevels();
					for (uint32_t j = 0; j < 6; ++j)
					{
						for (uint32_t k = 0; k < mipLevels; ++k)
						{
							GfxDevice::Copy(sourceTexture, perLightingSettingsData.reflectionProbeTexture, j, static_cast<uint32_t>(i * 6 + j), k);
						}
					}
				}
				perLightingSettingsData.reflectionProbesUpdateCount = reflectionProbesUpdateCount;
			}
			GfxDevice::SetGlobalTexture(s_ReflectionTextureId, perLightingSettingsData.reflectionProbeTexture == nullptr ? DefaultTextures::GetBlackCubeArray()->Get() : perLightingSettingsData.reflectionProbeTexture);
		}
	}

	void BakedLighting::ReleaseIfNeeded()
	{
		float time = Time::GetTime();
		if (s_ReleaseResourcesTimer < time)
		{
			for (auto it = s_PerLightingSettingsData.begin(); it != s_PerLightingSettingsData.end();)
			{
				Scene* scene = Scene::Get(static_cast<uint32_t>(it->first));
				LightingSettings* settings = static_cast<LightingSettings*>(ObjectDB::GetObject(static_cast<ObjectId>(it->first >> 32)));
				if (scene == nullptr || settings == nullptr)
				{
					auto& data = it->second;
					if (data.scaleOffsetBuffer != nullptr)
					{
						delete data.scaleOffsetBuffer;
					}
					if (data.probeVolumeTexture0 != nullptr)
					{
						GfxTexturePool::Release(data.probeVolumeTexture0);
						GfxTexturePool::Release(data.probeVolumeTexture1);
						GfxTexturePool::Release(data.probeVolumeTexture2);
						GfxTexturePool::Release(data.probeVolumeTexture3);
						GfxTexturePool::Release(data.probeVolumeTexture4);
						GfxTexturePool::Release(data.probeVolumeTexture5);
					}
					it = s_PerLightingSettingsData.erase(it);
				}
				else
				{
					++it;
				}
			}
			s_ReleaseResourcesTimer = time + 5.0f;
		}
	}
}