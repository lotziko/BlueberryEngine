#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class ComputeShader;
	class GfxBuffer;
	class GfxTexture;
	class Texture2D;
	class Material;

	class PostProcessing
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Draw(GfxTexture* msaaColor, GfxTexture* color, GfxTexture* output, const Rectangle& viewport, const Vector2Int& size, const bool& simplified);

	private:
		static inline ComputeShader* s_ResolveMSAABloomShader = nullptr;
		static inline Material* s_BloomMaterial = nullptr;
		static inline GfxBuffer* s_ResolveMSAABloomData = nullptr;
		static inline GfxBuffer* s_PostProcessingData = nullptr;
		static inline GfxBuffer* s_BloomData = nullptr;
		static inline Texture2D* s_BlueNoiseLUT = nullptr;
		static inline Texture2D* s_BRDFIntegrationLUT = nullptr;
	};
}