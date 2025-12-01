#pragma once

namespace Blueberry
{
	struct CullingResults;
	class ComputeShader;
	class GfxTexture;

	class RealtimeLights
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void PrepareShadows(CullingResults& results);
		static void BindLights(CullingResults& results);
		static void CalculateClusters();

	private:
		static inline ComputeShader* s_ClusteringShader = nullptr;
		static inline GfxTexture* s_LightIndexTexture = nullptr;
	};
}