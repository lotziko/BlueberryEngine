#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxTexture;
	class Texture2D;

	class PostProcessing
	{
	public:
		static void Draw(GfxTexture* color, const Rectangle& viewport);

	private:
		static inline Texture2D* m_BlueNoiseLUT = nullptr;
	};
}