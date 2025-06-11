#include "VolumetricFog.h"

#include "..\..\Assets\AssetLoader.h"
#include "..\..\Core\Time.h"
#include "..\GfxDevice.h"
#include "..\RenderContext.h"
#include "..\Buffers\FogLightDataConstantBuffer.h"
#include "..\Buffers\FogViewDataConstantBuffer.h"
#include "ShadowAtlas.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	#define MAX_REALTIME_LIGHTS 128

	struct CONSTANTS
	{
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

	static Vector3Int s_FrustumVolumeSize = Vector3Int(128, 96, 128);
	static size_t s_InjectFogVolumeId = TO_HASH("_InjectFogVolume");
	static size_t s_PreviousFrameInjectFogVolumeId = TO_HASH("_PreviousFrameInjectFogVolume");
	static size_t s_InjectedFogVolumeId = TO_HASH("_InjectedFogVolume");
	static size_t s_ScatterFogVolumeId = TO_HASH("_ScatterFogVolume");
	static size_t s_VolumetricFogTextureId = TO_HASH("_VolumetricFogTexture");

	void VolumetricFog::CalculateFrustum(const CullingResults& results, const CameraData& data, ShadowAtlas* atlas)
	{
		if (s_VolumetricFogShader == nullptr)
		{
			s_VolumetricFogShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/VolumetricFog.compute"));
			
			TextureProperties textureProperties = {};

			textureProperties.width = s_FrustumVolumeSize.x;
			textureProperties.height = s_FrustumVolumeSize.y;
			textureProperties.depth = s_FrustumVolumeSize.z;
			textureProperties.antiAliasing = 1;
			textureProperties.mipCount = 1;
			textureProperties.format = TextureFormat::R8G8B8A8_UNorm;
			textureProperties.dimension = TextureDimension::Texture3D;
			textureProperties.wrapMode = WrapMode::Clamp;
			textureProperties.filterMode = FilterMode::Linear;
			textureProperties.isRenderTarget = false;
			textureProperties.isReadable = false;
			textureProperties.isWritable = false;
			textureProperties.isUnorderedAccess = true;
			
			GfxDevice::CreateTexture(textureProperties, s_FrustumInjectVolume0);
			GfxDevice::CreateTexture(textureProperties, s_FrustumInjectVolume1);
			textureProperties.format = TextureFormat::R16G16B16A16_UNorm;
			GfxDevice::CreateTexture(textureProperties, s_FrustumScatterVolume);
		}

		List<Light*> lights;
		for (Light* light : results.lights)
		{
			if (light->IsCastingFog())
			{
				lights.emplace_back(light);
			}
		}
		bool isEven = Time::GetFrameCount() % 2 == 0;
		FogLightDataConstantBuffer::BindData(lights, atlas->GetSize());
		FogViewDataConstantBuffer::BindData(data, s_FrustumVolumeSize);
		GfxDevice::SetGlobalTexture(s_InjectFogVolumeId, isEven ? s_FrustumInjectVolume0 : s_FrustumInjectVolume1);
		GfxDevice::SetGlobalTexture(s_PreviousFrameInjectFogVolumeId, isEven ? s_FrustumInjectVolume1 : s_FrustumInjectVolume0);
		GfxDevice::Dispatch(s_VolumetricFogShader->GetKernel(0), s_FrustumVolumeSize.x / 8, s_FrustumVolumeSize.y / 8, 1);
		GfxDevice::SetGlobalTexture(s_InjectedFogVolumeId, isEven ? s_FrustumInjectVolume0 : s_FrustumInjectVolume1);
		GfxDevice::SetGlobalTexture(s_ScatterFogVolumeId, s_FrustumScatterVolume);
		GfxDevice::Dispatch(s_VolumetricFogShader->GetKernel(1), s_FrustumVolumeSize.x / 8, s_FrustumVolumeSize.y / 8, 1);
		GfxDevice::SetGlobalTexture(s_VolumetricFogTextureId, s_FrustumScatterVolume);
	}
}
