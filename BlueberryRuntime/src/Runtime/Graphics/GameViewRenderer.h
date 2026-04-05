#pragma once

namespace Blueberry
{
	class Scene;
	class GfxTexture;

	class GameViewRenderer
	{
	public:
		static void Draw(Scene* scene);

	private:
		static GfxTexture* s_RenderTarget;
	};
}