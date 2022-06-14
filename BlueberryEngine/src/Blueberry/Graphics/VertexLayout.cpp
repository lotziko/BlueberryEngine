#include "bbpch.h"
#include "VertexLayout.h"

namespace Blueberry
{
	VertexLayout::Element::Element(ElementType type, UINT offset) : m_Type(type), m_Offset(offset)
	{
	}

	constexpr UINT VertexLayout::Element::GetSize(ElementType type)
	{
		switch (type)
		{
		case ElementType::Position2D:
			return sizeof(Vector2);
		case ElementType::Position3D:
			return sizeof(Vector3);
		case ElementType::Float3Color:
			return sizeof(Vector3);
		case ElementType::Float4Color:
			return sizeof(Color);
		case ElementType::TextureCoord:
			return sizeof(Vector2);
		}

		BB_ERROR("Unknown vertex layout element type.");
		return 0;
	}

	VertexLayout::ElementType VertexLayout::Element::GetType() const
	{
		return m_Type;
	}

	UINT VertexLayout::Element::GetOffset() const
	{
		return m_Offset;
	}

	UINT VertexLayout::Element::GetOffsetAfter() const
	{
		return m_Offset + GetSize();
	}

	UINT VertexLayout::Element::GetSize() const
	{
		return GetSize(m_Type);
	}

	VertexLayout& VertexLayout::Append(ElementType type)
	{
		m_Elements.emplace_back(type, GetSize());
		return *this;
	}

	const VertexLayout::Element& VertexLayout::ResolveByIndex(UINT i) const
	{
		return m_Elements[i];
	}

	UINT VertexLayout::GetSize() const
	{
		return m_Elements.empty() ? 0 : m_Elements.back().GetOffsetAfter();
	}
}