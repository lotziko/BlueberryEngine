#include "RealtimeLights.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "..\RenderContext.h"
#include "..\Buffers\PerCameraLightDataConstantBuffer.h"
#include "ShadowAtlas.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	#define CLUSTERS_X 30
	#define CLUSTERS_Y 17
	#define CLUSTERS_Z 16
	#define MAX_LIGHTS 64
	#define LIGHT_TYPE_COUNT 2

	static size_t s_ClusteringLightIndexTextureId = TO_HASH("_ClusteringLightIndexTexture");
	static size_t s_LightIndexTextureId = TO_HASH("_LightIndexTexture");

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

	void RealtimeLights::Initialize()
	{
		s_ClusteringShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/Clustering.compute"));
		
		TextureProperties textureProperties = {};

		textureProperties.width = CLUSTERS_X * MAX_LIGHTS;
		textureProperties.height = CLUSTERS_Y * CLUSTERS_Z * LIGHT_TYPE_COUNT;
		textureProperties.depth = 1;
		textureProperties.antiAliasing = 1;
		textureProperties.mipCount = 1;
		textureProperties.format = TextureFormat::R32_UInt;
		textureProperties.dimension = TextureDimension::Texture2D;
		textureProperties.wrapMode = WrapMode::Clamp;
		textureProperties.filterMode = FilterMode::Point;
		textureProperties.isRenderTarget = false;
		textureProperties.isReadable = false;
		textureProperties.isWritable = false;
		textureProperties.isUnorderedAccess = true;
		
		GfxDevice::CreateTexture(textureProperties, s_LightIndexTexture);
	}

	void RealtimeLights::Shutdown()
	{
		Object::Destroy(s_ClusteringShader);
		delete s_LightIndexTexture;
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
		Light* mainLight = nullptr;
		List<Light*> lights = {};
		for (Light* light : results.lights)
		{
			if (light->GetType() == LightType::Directional)
			{
				mainLight = light;
				continue;
			}
			lights.emplace_back(light);
		}
		PerCameraLightDataConstantBuffer::BindData(results.camera, mainLight, results.skyRenderer, lights, atlas->GetSize());
	}

	void RealtimeLights::CalculateClusters()
	{
		GfxDevice::SetGlobalTexture(s_ClusteringLightIndexTextureId, s_LightIndexTexture);
		GfxDevice::Dispatch(s_ClusteringShader->GetKernel(0), CLUSTERS_X, CLUSTERS_Y, CLUSTERS_Z);
		GfxDevice::SetGlobalTexture(s_LightIndexTextureId, s_LightIndexTexture);
	}
}
