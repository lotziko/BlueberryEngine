#pragma once

#include "Blueberry\Core\Base.h"
#include "Enums.h"
#include <cmath>

namespace Blueberry
{
	class GfxBuffer;

	struct BB_API TextureProperties
	{
		uint32_t width;
		uint32_t height;
		uint32_t depth;
		const void* data;
		size_t dataSize;
		uint32_t antiAliasing;
		uint32_t mipCount;
		TextureFormat format;
		TextureDimension dimension;
		WrapMode wrapMode;
		FilterMode filterMode;
		uint8_t slices;
		TextureUsageFlags usageFlags;
	};

	struct BB_API BufferProperties
	{
		uint32_t elementSize;
		uint32_t elementCount;
		void* data;
		size_t dataSize;
		BufferFormat format;
		BufferUsageFlags usageFlags;
	};

	struct BB_API BottomLevelAccelerationStructureSubMesh
	{
		uint32_t indexStart;
		uint32_t indexCount;
		bool isOpaque;
	};

	struct BB_API BottomLevelAccelerationStructureProperties
	{
		GfxBuffer* vertexBuffer;
		GfxBuffer* indexBuffer;
		uint32_t vertexStride;
		uint32_t normalOffset;
		uint32_t tangentOffset;
		uint32_t uv0Offset;

		BottomLevelAccelerationStructureSubMesh subMeshes[16];
		uint32_t subMeshCount;
	};
}