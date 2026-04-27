#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class ComputeShader;
	class GfxBuffer;
	class GfxTexture;
	class Texture2D;
	class Material;
	class Camera;
	enum class CameraType;

	class PostProcessing
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Draw(Camera* camera, GfxTexture* msaaColor, GfxTexture* color, GfxTexture* output, const Rectangle& viewport, const Vector2Int& size, const CameraType& cameraType);

	private:
		static ComputeShader* s_ResolveMSAABloomShader;
		static Material* s_BloomMaterial;
		static GfxBuffer* s_ResolveMSAABloomData;
		static GfxBuffer* s_PostProcessingData;
		static GfxBuffer* s_BloomData;
		static Texture2D* s_BlueNoiseLUT;
		static Texture2D* s_BRDFIntegrationLUT;
	};
}