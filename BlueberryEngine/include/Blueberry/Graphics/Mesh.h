#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\DataList.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\VertexLayout.h"

namespace Blueberry
{
	class GfxVertexBuffer;
	class GfxIndexBuffer;

	class BB_API SubMeshData : public Data
	{
		DATA_DECLARATION(SubMeshData)

	public:
		const uint32_t& GetIndexStart() const;
		void SetIndexStart(const uint32_t& indexStart);

		const uint32_t& GetIndexCount() const;
		void SetIndexCount(const uint32_t& indexCount);

	private:
		uint32_t m_IndexStart;
		uint32_t m_IndexCount;
	};

	class BB_API Mesh : public Object
	{
		OBJECT_DECLARATION(Mesh)

	public:
		Mesh() = default;
		virtual ~Mesh();

		const uint32_t& GetVertexCount();
		const uint32_t& GetIndexCount();
		const uint32_t GetSubMeshCount();
		const SubMeshData& GetSubMesh(const uint32_t& index);
		
		const List<Vector3>& GetVertices();
		const List<uint32_t>& GetIndices();

		void SetVertices(const Vector3* vertices, const uint32_t& vertexCount);
		void SetNormals(const Vector3* normals, const uint32_t& vertexCount);
		void SetTangents(const Vector4* tangents, const uint32_t& vertexCount);
		void SetColors(const Color* colors, const uint32_t& vertexCount);
		void SetIndices(const uint32_t* indices, const uint32_t& indexCount);
		void SetUVs(const int& channel, const Vector2* uvs, const uint32_t& uvCount);
		void SetSubMesh(const uint32_t& index, const SubMeshData& data);

		void GenerateTangents();

		const Topology& GetTopology();
		void SetTopology(const Topology& topology);

		const AABB& GetBounds();

		void Apply();
		const VertexLayout& GetLayout();

		static Mesh* Create();

	private:
		GfxVertexBuffer* m_VertexBuffer;
		GfxIndexBuffer* m_IndexBuffer;
		bool m_BufferIsDirty = false;

		List<Vector3> m_Vertices;
		List<Vector3> m_Normals;
		List<Vector4> m_Tangents;
		List<Color> m_Colors;
		List<uint32_t> m_Indices;
		List<Vector2> m_UVs[4] = {};

		List<float> m_VertexData;
		List<uint32_t> m_IndexData;
		DataList<SubMeshData> m_SubMeshes;
		VertexLayout m_Layout;

		uint32_t m_VertexCount;
		uint32_t m_IndexCount;
		AABB m_Bounds;

		Topology m_Topology = Topology::TriangleList;

		friend struct GfxDrawingOperation;
		friend class TangentGenerator;
	};
}