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
		static inline GfxTexture* s_RenderTargetTexture = nullptr;
		static inline GfxTexture* s_CubeTexture = nullptr;
		static inline Camera* s_Camera = nullptr;
	};
}