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
		UINT mipCount;
		TextureFormat format;
		WrapMode wrapMode;
		FilterMode filterMode;

		static UINT GetMipCount(const UINT& width, const UINT& height, const bool& generateMips)
		{
			if (generateMips)
			{
				UINT mipCount = (UINT)log2(Max((float)width, (float)height));
				// Based on https://stackoverflow.com/questions/108318/how-can-i-test-whether-a-number-is-a-power-of-2
				if ((width & (width - 1)) == 0 && (height & (height - 1)) == 0)
				{
					return mipCount;
				}
			}
			return 1;
		}
	};
}