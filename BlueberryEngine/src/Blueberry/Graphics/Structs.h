#pragma once

namespace Blueberry
{
	struct TextureProperties
	{
		UINT width;
		UINT height;
		void* data;
		size_t dataSize;
		bool isRenderTarget;
	};

	struct ByteData
	{
		BYTE* data;
		size_t size;
	};
}