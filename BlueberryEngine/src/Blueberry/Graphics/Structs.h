#pragma once
#include "Enums.h"

#include <cmath>

namespace Blueberry
{
	struct TextureProperties
	{
		uint32_t width;
		uint32_t height;
		void* data;
		size_t dataSize;
		bool isRenderTarget;
		bool isReadable;
		uint32_t antiAliasing;
		uint32_t mipCount;
		TextureFormat format;
		WrapMode wrapMode;
		FilterMode filterMode;
	};
}