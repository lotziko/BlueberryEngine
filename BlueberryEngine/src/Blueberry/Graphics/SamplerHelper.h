#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	class SamplerHelper
	{
	public:
		static bool ParseName(size_t nameHash, FilterMode& filterMode, WrapMode& wrapMode);
	};
}