#include "Blueberry\Graphics\VertexLayout.h"

#include "Blueberry\Tools\CRCHelper.h"

namespace Blueberry
{
	VertexLayout::Element::Element(uint32_t size) : m_Size(size)
	{
	}

	uint32_t VertexLayout::Element::GetSize() const
	{
		return m_Size;
	}

	uint32_t VertexLayout::Element::GetOffset() const
	{
		return m_Size;
	}

	VertexLayout& VertexLayout::Append(VertexAttribute type, uint32_t size)
	{
		m_Elements[static_cast<uint32_t>(type)] = Element(size);
		m_Crc = UINT32_MAX;
		return *this;
	}

	VertexLayout& VertexLayout::Apply()
	{
		m_Crc = 0;
		uint32_t offset = 0;
		for (uint32_t i = 0; i < VERTEX_ATTRIBUTE_COUNT; ++i)
		{
			Element& element = m_Elements[i];
			if (element.m_Size > 0)
			{
				element.m_Offset = offset;
				offset += element.m_Size;
			}
			m_Crc = CRCHelper::Calculate(element.m_Offset, m_Crc);
		}
		m_Size = offset;
		return *this;
	}

	bool VertexLayout::Has(VertexAttribute type) const
	{
		return m_Elements[static_cast<uint32_t>(type)].m_Size > 0;
	}

	uint32_t VertexLayout::GetOffset(VertexAttribute type) const
	{
		return m_Elements[static_cast<uint32_t>(type)].m_Offset;
	}

	uint32_t VertexLayout::GetOffset(uint32_t index) const
	{
		return m_Elements[index].m_Offset;
	}

	uint32_t VertexLayout::GetSize(VertexAttribute type) const
	{
		return m_Elements[static_cast<uint32_t>(type)].m_Size;
	}

	uint32_t VertexLayout::GetSize() const
	{
		return m_Size;
	}

	uint32_t VertexLayout::GetCrc() const
	{
		return m_Crc;
	}
}