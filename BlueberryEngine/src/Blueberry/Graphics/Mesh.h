#pragma once
#include "Blueberry\Graphics\VertexLayout.h"

namespace Blueberry
{
	class GfxVertexBuffer;
	class GfxIndexBuffer;

	class Mesh : public Object
	{
		OBJECT_DECLARATION(Mesh)

	public:
		Mesh() = default;
		Mesh(const VertexLayout& layout, const UINT& vertexCount, const UINT& indexCount);

		const UINT& GetVertexCount();
		const UINT& GetIndexCount();
		void SetVertexData(float* data, const UINT& vertexCount);
		void SetIndexData(UINT* data, const UINT& indexCount);

		static Ref<Mesh> Create(const VertexLayout& layout, const UINT& vertexCount, const UINT& indexCount);

	private:
		Ref<GfxVertexBuffer> m_VertexBuffer;
		Ref<GfxIndexBuffer> m_IndexBuffer;

		UINT m_VertexCount;
		UINT m_IndexCount;

		friend struct GfxDrawingOperation;
	};
}