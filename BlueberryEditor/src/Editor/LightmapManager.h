#pragma once

namespace Blueberry
{
	class GfxTexture;

	class LightmapManager
	{
	public:
		static void Initialize();
		static void Bake();

		static GfxTexture* GetTexture();
	};
}