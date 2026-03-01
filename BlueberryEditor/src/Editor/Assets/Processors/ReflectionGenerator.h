#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class TextureCube;
	class GfxTexture;
	class Camera;
	class Material;
	class Scene;
	class SkyRenderer;
	class ReflectionProbe;

	class ReflectionGenerator
	{
	public:
		static void GenerateReflectionTexture(SkyRenderer* skyRenderer);
		static void GenerateReflectionTexture(ReflectionProbe* reflectionProbe);

	private:
		static void Initialize();
		static TextureCube* Save(const uint32_t& index);

	private:
		static GfxTexture* s_RenderTargetTexture;
		static GfxTexture* s_CubeTexture;
		static Camera* s_Camera;
	};
}