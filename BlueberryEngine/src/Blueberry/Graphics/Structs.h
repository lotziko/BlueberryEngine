#pragma once
#include "Enums.h"

#include <cmath>

namespace Blueberry
{
	struct TextureProperties
	{
		UINT width;
		UINT height;
		void* data;
		size_t dataSize;
		bool isRenderTarget;
		bool isReadable;
		UINT antiAliasing;
		UINT mipCount;
		TextureFormat format;
		WrapMode wrapMode;
		FilterMode filterMode;
	};
}