#pragma once

namespace Blueberry
{
	struct CullingResults;
	class ShadowAtlas;
	class ComputeShader;
	class GfxTexture;

	class RealtimeLights
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void PrepareShadows(CullingResults& results, ShadowAtlas* atlas);
		static void BindLights(CullingResults& results, ShadowAtlas* atlas);
		static void CalculateClusters();

	private:
		static inline ComputeShader* s_ClusteringShader = nullptr;
		static inline GfxTexture* s_LightIndexTexture = nullptr;
		static inline GfxTexture* s_DebugTexture = nullptr;
	};
}