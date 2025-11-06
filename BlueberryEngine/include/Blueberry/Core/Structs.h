#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	using ByteData = Blueberry::List<uint8_t>;

	struct ObjectPtrData
	{
		FileId fileId;
		Guid guid;
	};
}