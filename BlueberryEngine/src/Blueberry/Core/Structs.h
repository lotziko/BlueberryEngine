#pragma once
#include "Base.h"
#include "Guid.h"

namespace Blueberry
{
	struct ByteData
	{
		byte* data;
		size_t size;
	};

	struct ObjectPtrData
	{
		uint64_t fileId;
		bool isAsset;
		Guid guid;
	};
}