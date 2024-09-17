#pragma once
#include "Blueberry\Graphics\VertexLayout.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Core\DataPtr.h"

namespace Blueberry
{
	class GfxVertexBuffer;
	class GfxIndexBuffer;

	class SubMeshData : public Data
	{
		DATA_DECLARATION(SubMeshData)

	public:
		const UINT& GetIndexStart();
		void SetIndexStart(const UINT& indexStart);

		const UINT& GetIndexCount();
		void SetIndexCount(const UINT& indexCount);

		static void BindProperties();

	private:
		UINT m_IndexStart;
		UINT m_IndexCount;
	};

	class Mesh : public Object
	{
		OBJECT_DECLARATION(Mesh)

	public:
		Mesh() = default;
		virtual ~Mesh();

		const UINT& GetVertexCount();
		const UINT& GetIndexCount();
		const UINT& GetSubMeshCount();
		SubMeshData* GetSubMesh(const UINT& index);
		
		void SetVertices(const Vector3* vertices, const UINT& vertexCount);
		void SetNormals(const Vector3* normals, const UINT& vertexCount);
		void SetTangents(const Vector4* tangents, const UINT& vertexCount);
		void SetIndices(const UINT* indices, const UINT& indexCount);
		void SetUVs(const int& channel, const Vector2* uvs, const UINT& uvCount);
		void SetSubMesh(const UINT& index, SubMeshData* data);

		void GenerateTangents();

		const Topology& GetTopology();
		void SetTopology(const Topology& topology);

		const AABB& GetBounds();

		void Apply();

		static Mesh* Create();

		static void BindProperties();

	private:
		VertexLayout GetLayout();

	private:
		GfxVertexBuffer* m_VertexBuffer;
		GfxIndexBuffer* m_IndexBuffer;
		bool m_BufferIsDirty = false;

		std::vector<Vector3> m_Vertices;
		std::vector<Vector3> m_Normals;
		std::vector<Vector4> m_Tangents;
		std::vector<UINT> m_Indices;
		std::vector<Vector2> m_UVs[8] = {};

		std::vector<float> m_VertexData;
		std::vector<UINT> m_IndexData;
		std::vector<DataPtr<SubMeshData>> m_SubMeshes;

		UINT m_VertexCount;
		UINT m_IndexCount;
		UINT m_ChannelFlags;
		AABB m_Bounds;

		Topology m_Topology = Topology::TriangleList;

		friend struct GfxDrawingOperation;
		friend class TangentGenerator;
	};
}