#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class ComputeShader;
	class GfxTexture;
	struct CullingResults;
	struct CameraData;

	class VolumetricFog
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void CalculateFrustum(const CullingResults& results, const CameraData& data);
		static GfxTexture* GetFrustumTexture();

	private:
		static ComputeShader* s_VolumetricFogShader;
		static GfxTexture* s_FrustumVolume0;
		static GfxTexture* s_FrustumVolume1;
		static GfxTexture* s_FrustumVolume2;
		static ObjectId s_CameraId;
	};
}