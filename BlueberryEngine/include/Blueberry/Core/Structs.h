#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	struct ObjectPtrData
	{
		FileId fileId;
		Guid guid;
	};

	template <typename T>
	struct DataWrapper
	{
		T& reference;
	};
}