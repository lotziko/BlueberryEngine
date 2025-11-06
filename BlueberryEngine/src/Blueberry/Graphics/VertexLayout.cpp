#include "Blueberry\Graphics\VertexLayout.h"

#include "Blueberry\Tools\CRCHelper.h"

namespace Blueberry
{
	VertexLayout::Element::Element(const uint32_t& size) : m_Size(size)
	{
	}

	const uint32_t& VertexLayout::Element::GetSize()
	{
		return m_Size;
	}

	const uint32_t& VertexLayout::Element::GetOffset()
	{
		return m_Size;
	}

	VertexLayout& VertexLayout::Append(const VertexAttribute& type, const uint32_t& size)
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

	const bool VertexLayout::Has(const VertexAttribute& type)
	{
		return m_Elements[static_cast<uint32_t>(type)].m_Size > 0;
	}

	const uint32_t& VertexLayout::GetOffset(const VertexAttribute& type)
	{
		return m_Elements[static_cast<uint32_t>(type)].m_Offset;
	}

	const uint32_t& VertexLayout::GetOffset(const uint32_t& index)
	{
		return m_Elements[index].m_Offset;
	}

	const uint32_t& VertexLayout::GetSize(const VertexAttribute& type)
	{
		return m_Elements[static_cast<uint32_t>(type)].m_Size;
	}

	const uint32_t& VertexLayout::GetSize()
	{
		return m_Size;
	}

	const uint32_t& VertexLayout::GetCrc()
	{
		return m_Crc;
	}
}