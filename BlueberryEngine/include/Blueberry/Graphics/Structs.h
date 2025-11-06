#pragma once

#include "Blueberry\Core\Base.h"
#include "Enums.h"
#include <cmath>

namespace Blueberry
{
	struct BB_API TextureProperties
	{
		uint32_t width;
		uint32_t height;
		uint32_t depth;
		void* data;
		size_t dataSize;
		bool isRenderTarget;
		bool isReadable;
		bool isWritable;
		bool isUnorderedAccess;
		uint32_t antiAliasing;
		uint32_t mipCount;
		TextureFormat format;
		TextureDimension dimension;
		WrapMode wrapMode;
		FilterMode filterMode;
		uint8_t slices;
		bool generateMipMaps;
	};

	struct BB_API BufferProperties
	{
		BufferType type;
		uint32_t elementSize;
		uint32_t elementCount;
		void* data;
		size_t dataSize;
		bool isReadable;
		bool isWritable;
		bool isUnorderedAccess;
		BufferFormat format;
	};
}