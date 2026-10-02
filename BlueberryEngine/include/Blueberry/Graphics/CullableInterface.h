#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class BB_API CullableInterface
	{
	public:
		virtual void OnPreCull() = 0;
	};
}