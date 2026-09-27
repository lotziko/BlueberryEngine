#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\GfxTexturePool.h"

namespace Blueberry
{
	class ComputeShader;
	class GfxTexture;
	struct CullingResults;
	struct CameraData;
	class PerCameraData;

	struct PerCameraVolumetricFogData
	{
		std::unique_ptr<GfxTexture, ReturnTextureToPool> frustumVolume0;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> frustumVolume1;
		std::unique_ptr<GfxTexture, ReturnTextureToPool> frustumVolume2;
	};

	class VolumetricFog
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void CalculateFrustum(const CullingResults& results, const CameraData& data, PerCameraData& perCameraData);
		static GfxTexture* GetFrustumTexture(const PerCameraData& perCameraData);

	private:
		static ComputeShader* s_VolumetricFogShader;
	};
}