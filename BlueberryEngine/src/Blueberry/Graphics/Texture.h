#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Texture : public Object
	{
		OBJECT_DECLARATION(Texture)

	public:
		Texture() = default;
		virtual ~Texture() = default;

		virtual void* GetHandle() = 0;
	};
}