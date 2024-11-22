#pragma once

#include <vector>

namespace Blueberry
{
	class VertexLayout
	{
	public:
		enum ElementType
		{
			Position2D,
			Position3D,
			Normal,
			Tangent,
			Float3Color,
			Float4Color,
			TextureCoord,
			Index,
		};
		class Element
		{
		public:
			Element(ElementType type, uint32_t offset);

			static constexpr uint32_t GetSize(ElementType type);

			ElementType GetType() const;
			uint32_t GetOffset() const;
			uint32_t GetOffsetAfter() const;
			uint32_t GetSize() const;

		private:
			ElementType m_Type;
			uint32_t m_Offset;
		};
	public:
		template<ElementType T>
		const Element& Resolve() const;

		VertexLayout& Append(ElementType type);

		const Element& ResolveByIndex(uint32_t i) const;
		uint32_t GetSize() const;

	private:
		std::vector<Element> m_Elements;
	};

	template<VertexLayout::ElementType T>
	inline const VertexLayout::Element& VertexLayout::Resolve() const
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
}