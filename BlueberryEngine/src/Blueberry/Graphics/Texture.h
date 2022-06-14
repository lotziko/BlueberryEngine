#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Texture : public Object
	{
	public:
		Texture() = default;
		virtual ~Texture() = default;

		virtual void* GetHandle() = 0;

		virtual void Bind() const = 0;
	};
}