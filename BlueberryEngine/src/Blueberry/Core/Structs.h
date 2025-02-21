#pragma once
#include "Base.h"
#include "Guid.h"

namespace Blueberry
{
	struct ByteData
	{
		uint8_t* data = nullptr;
		size_t size = 0;
	};

	struct ObjectPtrData
	{
		FileId fileId;
		Guid guid;
	};
}