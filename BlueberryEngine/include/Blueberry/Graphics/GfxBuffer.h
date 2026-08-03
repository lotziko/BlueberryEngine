#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual ~GfxBuffer() = default;

		uint32_t GetElementSize() const;
		uint32_t GetElementCount() const;

		uint32_t GetIndex() const;

		virtual void GetData(void* data) = 0;
		virtual void SetData(const void* data, size_t size) = 0;

	protected:
		uint32_t m_ElementSize = 0;
		uint32_t m_ElementCount = 0;
		bool m_IsConstant = false;
		uint32_t m_Index = 0;
	};
}