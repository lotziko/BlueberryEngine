#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual ~GfxBuffer() = default;

		virtual void* Map() = 0;
		virtual void Unmap() = 0;

		virtual void GetData(void* data) = 0;
		virtual void SetData(const void* data, size_t size) = 0;

		virtual uint32_t GetElementSize() const = 0;
		virtual uint32_t GetElementCount() const = 0;

	protected:
		uint32_t m_Index;
	};
}