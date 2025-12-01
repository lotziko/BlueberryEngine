#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class GfxTexture;

	class DefaultRenderer
	{
	public:
		static void Initialize();
		static void Shutdown();
		
		static void Draw(Scene* scene, Camera* camera, Rectangle viewport, Color background, GfxTexture* colorOutput = nullptr, GfxTexture* depthOutput = nullptr);
	};
}