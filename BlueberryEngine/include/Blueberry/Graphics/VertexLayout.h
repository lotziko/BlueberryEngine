#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	enum class VertexAttribute
	{
		Position,
		Normal,
		Tangent,
		Color,
		Texcoord0,
		Texcoord1,
		Texcoord2,
		Texcoord4,
	};

	static const uint32_t VERTEX_ATTRIBUTE_COUNT = 8;

	class VertexLayout
	{
	public:
		class Element
		{
		public:
			Element() = default;
			Element(const uint32_t& size);

			const uint32_t& GetSize();
			const uint32_t& GetOffset();

		private:
			uint32_t m_Offset = 0;
			uint32_t m_Size = 0;

			friend class VertexLayout;
		};
	public:
		VertexLayout() = default;
		virtual ~VertexLayout() = default;
		
		VertexLayout& Append(const VertexAttribute& type, const uint32_t& size);
		VertexLayout& Apply();
		const bool Has(const VertexAttribute& type);
		const uint32_t& GetOffset(const VertexAttribute& type);
		const uint32_t& GetOffset(const uint32_t& index);
		const uint32_t& GetSize(const VertexAttribute& type);
		const uint32_t& GetSize();
		const uint32_t& GetCrc();

	private:
		Element m_Elements[VERTEX_ATTRIBUTE_COUNT];
		uint32_t m_Size = 0;
		uint32_t m_Crc = UINT32_MAX;
	};
}