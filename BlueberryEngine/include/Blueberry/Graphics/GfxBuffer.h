#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual ~GfxBuffer() = default;

		virtual void GetData(void* data) = 0;
		virtual void SetData(const void* data, const uint32_t& size) = 0;

		virtual const uint32_t& GetElementSize() = 0;
	};
}