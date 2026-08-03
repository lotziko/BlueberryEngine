#include "Blueberry\Graphics\GfxBuffer.h"

namespace Blueberry
{
	uint32_t GfxBuffer::GetElementSize() const
	{
		return m_ElementSize;
	}

	uint32_t GfxBuffer::GetElementCount() const
	{
		return m_ElementCount;
	}

	uint32_t GfxBuffer::GetIndex() const
	{
		return m_Index;
	}
}