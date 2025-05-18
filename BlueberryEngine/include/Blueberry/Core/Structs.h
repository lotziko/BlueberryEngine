#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"

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