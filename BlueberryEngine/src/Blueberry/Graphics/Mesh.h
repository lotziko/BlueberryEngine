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
		virtual ~Mesh();

		const UINT& GetVertexCount();
		const UINT& GetIndexCount();
		
		void SetVertices(const Vector3* vertices, const UINT& vertexCount);
		void SetIndices(const UINT* indices, const UINT& indexCount);
		void SetUVs(const int& channel, const Vector2* uvs, const UINT& uvCount);

		const Topology& GetTopology();
		void SetTopology(const Topology& topology);

		void Apply();

		static Mesh* Create();

		static void BindProperties();

	private:
		GfxVertexBuffer* m_VertexBuffer;
		GfxIndexBuffer* m_IndexBuffer;

		Vector3* m_Vertices;
		UINT* m_Indices;
		Vector2* m_UVs[8] = {};

		float* m_VertexData = nullptr;
		UINT m_VertexDataSize = 0;

		UINT m_VertexCount;
		UINT m_IndexCount;

		Topology m_Topology = Topology::TriangleList;

		friend struct GfxDrawingOperation;
	};
}