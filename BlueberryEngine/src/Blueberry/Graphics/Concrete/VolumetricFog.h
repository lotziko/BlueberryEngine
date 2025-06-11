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
		static void CalculateFrustum(const CullingResults& results, const CameraData& data, ShadowAtlas* atlas);

	private:
		static inline ComputeShader* s_VolumetricFogShader = nullptr;
		static inline GfxTexture* s_FrustumInjectVolume0 = nullptr;
		static inline GfxTexture* s_FrustumInjectVolume1 = nullptr;
		static inline GfxTexture* s_FrustumScatterVolume = nullptr;
	};
}