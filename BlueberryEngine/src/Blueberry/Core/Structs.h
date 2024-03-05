#pragma once
#include "Base.h"
#include "Guid.h"

namespace Blueberry
{
	struct ByteData
	{
		byte* data = nullptr;
		size_t size = 0;
	};

	struct ObjectPtrData
	{
		FileId fileId;
		bool isAsset;
		Guid guid;
	};
}