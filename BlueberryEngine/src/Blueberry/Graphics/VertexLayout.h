#pragma once

class VertexLayout
{
public:
	enum ElementType
	{
		Position2D,
		Position3D,
		Float3Color,
		Float4Color,
		TextureCoord,
	};
	class Element
	{
	public:
		Element(ElementType type, UINT offset);

		static constexpr UINT GetSize(ElementType type);

		ElementType GetType() const;
		UINT GetOffset() const;
		UINT GetOffsetAfter() const;
		UINT GetSize() const;

	private:
		ElementType m_Type;
		UINT m_Offset;
	};
public:
	template<ElementType T>
	const Element& Resolve() const
	{
		for (auto& element : m_Elements)
		{
			if (element.GetType() == T)
			{
				return element;
			}
			BB_ERROR("Could not resolve element type.");
			return m_Elements.front();
		}
	}

	VertexLayout& Append(ElementType type);

	const Element& ResolveByIndex(UINT i) const;
	UINT GetSize() const;

private:
	std::vector<Element> m_Elements;
};