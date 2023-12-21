#pragma once
#include "Blueberry\Graphics\VertexLayout.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	class GfxVertexBuffer;
	class GfxIndexBuffer;

	class Mesh : public Object
	{
		OBJECT_DECLARATION(Mesh)

	public:
		Mesh() = default;

		const UINT& GetVertexCount();
		const UINT& GetIndexCount();
		void SetVertexData(float* data, const UINT& vertexCount);
		void SetIndexData(UINT* data, const UINT& indexCount);

		const Topology& GetTopology();
		void SetTopology(const Topology& topology);

		static Mesh* Create(const VertexLayout& layout, const UINT& vertexCount, const UINT& indexCount);

		static void BindProperties();

	private:
		GfxVertexBuffer* m_VertexBuffer;
		GfxIndexBuffer* m_IndexBuffer;

		UINT m_VertexCount;
		UINT m_IndexCount;

		Topology m_Topology = Topology::TriangleList;

		friend struct GfxDrawingOperation;
	};
}