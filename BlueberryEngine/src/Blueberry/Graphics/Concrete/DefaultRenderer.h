#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class GfxTexture;
	class Material;
	class HBAORenderer;
	class ShadowAtlas;
	class AutoExposure;

	class TextureCube;

	class DefaultRenderer
	{
	public:
		static void Initialize();
		static void Shutdown();
		
		static void Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, GfxTexture* colorOutput = nullptr, GfxTexture* depthOutput = nullptr, const bool& simplified = false);
		
	private:
		static inline Material* s_ResolveMSAAMaterial = nullptr;
		static inline ShadowAtlas* s_ShadowAtlas = nullptr;
	};
}