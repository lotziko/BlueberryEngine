#pragma once

namespace Blueberry
{
	class ComputeShader;
	class GfxTexture;
	struct CullingResults;
	struct CameraData;
	class ShadowAtlas;

	class VolumetricFog
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void CalculateFrustum(const CullingResults& results, const CameraData& data, ShadowAtlas* atlas);
		static GfxTexture* GetFrustumTexture();

	private:
		static inline ComputeShader* s_VolumetricFogShader = nullptr;
		static inline GfxTexture* s_FrustumVolume0 = nullptr;
		static inline GfxTexture* s_FrustumVolume1 = nullptr;
	};
}