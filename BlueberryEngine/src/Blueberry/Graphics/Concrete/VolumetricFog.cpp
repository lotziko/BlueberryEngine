#include "VolumetricFog.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "..\RenderContext.h"
#include "..\Buffers\FogViewDataConstantBuffer.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "PerCameraData.h"

namespace Blueberry
{
	ComputeShader* VolumetricFog::s_VolumetricFogShader = nullptr;

	static Vector3Int s_FrustumVolumeSize = Vector3Int(128, 96, 128);
	static size_t s_InjectFogVolumeId = TO_HASH("_InjectFogVolume");
	static size_t s_InjectedFogVolumeId = TO_HASH("_InjectedFogVolume");
	static size_t s_PreviousFrameInjectFogVolumeId = TO_HASH("_PreviousFrameInjectFogVolume");
	static size_t s_ScatterFogVolumeId = TO_HASH("_ScatterFogVolume");

	void VolumetricFog::Initialize()
	{
		s_VolumetricFogShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/VolumetricFog.compute"));
	}

	void VolumetricFog::Shutdown()
	{
		Object::Destroy(s_VolumetricFogShader);
	}

	void VolumetricFog::CalculateFrustum(const CullingResults& results, const CameraData& data, PerCameraData& perCameraData)
	{
		PerCameraVolumetricFogData& volumetricFogData = perCameraData.m_VolumetricFogData;

		if (volumetricFogData.frustumVolume0 == nullptr)
		{
			TextureProperties textureProperties = {};
			textureProperties.width = s_FrustumVolumeSize.x;
			textureProperties.height = s_FrustumVolumeSize.y;
			textureProperties.depth = s_FrustumVolumeSize.z;
			textureProperties.antiAliasing = 1;
			textureProperties.mipCount = 1;
			textureProperties.format = TextureFormat::R16G16B16A16_Float;
			textureProperties.dimension = TextureDimension::Texture3D;
			textureProperties.wrapMode = WrapMode::Clamp;
			textureProperties.filterMode = FilterMode::Bilinear;
			textureProperties.usageFlags = TextureUsageFlags::RenderTarget | TextureUsageFlags::UnorderedAccess;
			
			volumetricFogData.frustumVolume0.reset(GfxTexturePool::Get(textureProperties));
			volumetricFogData.frustumVolume1.reset(GfxTexturePool::Get(textureProperties));
			volumetricFogData.frustumVolume2.reset(GfxTexturePool::Get(textureProperties));
		}

		GfxTexture* frustumVolume0 = volumetricFogData.frustumVolume0.get();
		GfxTexture* frustumVolume1 = volumetricFogData.frustumVolume1.get();
		GfxTexture* frustumVolume2 = volumetricFogData.frustumVolume2.get();
		
		FogViewDataConstantBuffer::BindData(data, s_FrustumVolumeSize);
		GfxDevice::SetGlobalTexture(s_InjectFogVolumeId, frustumVolume0);
		GfxDevice::SetGlobalTexture(s_PreviousFrameInjectFogVolumeId, frustumVolume1);
		GfxDevice::Dispatch(s_VolumetricFogShader, 0, s_FrustumVolumeSize.x / 16, s_FrustumVolumeSize.y / 16, 1);
		
		GfxDevice::SetGlobalTexture(s_InjectedFogVolumeId, frustumVolume0);
		GfxDevice::SetGlobalTexture(s_ScatterFogVolumeId, frustumVolume2);
		GfxDevice::Dispatch(s_VolumetricFogShader, 1, s_FrustumVolumeSize.x / 8, s_FrustumVolumeSize.y / 8, 1);
		std::swap(volumetricFogData.frustumVolume0, volumetricFogData.frustumVolume1);
	}

	GfxTexture* VolumetricFog::GetFrustumTexture(const PerCameraData& perCameraData)
	{
		return perCameraData.m_VolumetricFogData.frustumVolume2.get();
	}
}
