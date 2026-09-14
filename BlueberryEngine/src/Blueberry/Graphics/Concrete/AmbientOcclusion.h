#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Camera;
	class ComputeShader;
	class GfxBuffer;
	class GfxTexture;

	class AmbientOcclusion
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Draw(Camera* camera, GfxTexture* depthStencil, GfxTexture* normals, GfxTexture* output, const Rectangle& viewport);
	
	private:
		static ComputeShader* s_GTAOShader;
		static GfxBuffer* s_GTAOData;
	};
}