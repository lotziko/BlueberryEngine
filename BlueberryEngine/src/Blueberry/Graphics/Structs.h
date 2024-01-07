#pragma once
#include "Enums.h"

namespace Blueberry
{
	struct TextureProperties
	{
		UINT width;
		UINT height;
		void* data;
		size_t dataSize;
		TextureType type;
	};
}