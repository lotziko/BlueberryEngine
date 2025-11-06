#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Object;
	class Texture2D;
	class GfxTexture;

	class ThumbnailRenderer
	{
	public:
		static bool CanDraw(const size_t& type);
		static bool Draw(unsigned char* output, const uint32_t& size, Object* asset);

	private:
		static inline GfxTexture* s_ThumbnailRenderTarget = nullptr;
	};
}