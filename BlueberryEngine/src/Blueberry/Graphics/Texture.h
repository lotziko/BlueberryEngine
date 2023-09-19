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

		virtual UINT GetWidth() const = 0;
		virtual UINT GetHeight() const = 0;
		virtual void* GetHandle() = 0;
	};
}