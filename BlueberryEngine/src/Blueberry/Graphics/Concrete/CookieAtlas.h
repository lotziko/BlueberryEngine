#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	struct CullingResults;
	class Texture;
	class GfxTexture;

	class CookieAtlas
	{
	public:
		static void Initialize();
		static void PrepareCookies(CullingResults& results);
		static GfxTexture* GetAtlasTexture();

	private:
		static uint32_t GetIndex(Texture* cookie);

	private:
		static inline GfxTexture* s_AtlasTexture = nullptr;
		static List<ObjectId> s_Cookies;
	};
}