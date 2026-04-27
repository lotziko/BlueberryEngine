#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\VertexLayout.h"

namespace Blueberry
{
	class GfxBuffer;

	class BB_API SubMeshData : public Data
	{
		DATA_DECLARATION(SubMeshData)

	public:
		uint32_t GetIndexStart() const;
		void SetIndexStart(uint32_t indexStart);

		uint32_t GetIndexCount() const;
		void SetIndexCount(uint32_t indexCount);

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

		uint32_t GetVertexCount() const;
		uint32_t GetIndexCount() const;
		size_t GetBindPoseCount() const;
		const Matrix& GetBindPose(size_t index) const;
		uint32_t GetSubMeshCount() const;
		const SubMeshData& GetSubMesh(size_t index) const;
		
		Vector3* GetVertices();
		Vector3* GetNormals();
		Vector4* GetTangents();
		Color* GetColors();
		Vector4* GetBoneWeights();
		Vector4Uint* GetBoneIndices();
		uint32_t* GetIndices();
		float* GetUVs(int channel);
		uint32_t GetUVSize(int channel);

		void SetVertices(const Vector3* vertices, uint32_t vertexCount);
		void SetNormals(const Vector3* normals, uint32_t vertexCount);
		void SetTangents(const Vector4* tangents, uint32_t vertexCount);
		void SetColors(const Color* colors, uint32_t vertexCount);
		void SetIndices(const uint32_t* indices, uint32_t indexCount);
		void SetUVs(const int& channel, const Vector2* uvs, uint32_t uvCount);
		void SetUVs(const int& channel, const Vector3* uvs, uint32_t uvCount);
		void SetUVs(const int& channel, const Vector4* uvs, uint32_t uvCount);
		void SetBoneWeights(const Vector4* weights, uint32_t vertexCount);
		void SetBoneIndices(const Vector4Uint* indices, uint32_t vertexCount);
		void SetBindPoses(const List<Matrix>& bindPoses);
		void SetSubMesh(uint32_t index, const SubMeshData& data);

		void GenerateTangents();

		Topology GetTopology() const;
		void SetTopology(Topology topology);

		const AABB& GetBounds() const;

		void Apply();

		GfxBuffer* GetVertexBuffer() const;
		GfxBuffer* GetIndexBuffer() const;
		const VertexLayout& GetLayout() const;

		uint32_t GetUpdateCount() const;

		static Mesh* Create();

	private:
		GfxBuffer* m_VertexBuffer = nullptr;
		GfxBuffer* m_IndexBuffer = nullptr;
		bool m_BufferIsDirty = false;

		List<Vector3> m_Vertices;
		List<Vector3> m_Normals;
		List<Vector4> m_Tangents;
		List<Color> m_Colors;
		List<float> m_UVs[4] = {};
		List<Vector4> m_BoneWeights;
		List<Vector4Uint> m_BoneIndices;

		List<float> m_VertexData;
		List<uint32_t> m_IndexData;
		List<Matrix> m_BindPoses;
		List<SubMeshData> m_SubMeshes;
		VertexLayout m_Layout;

		uint32_t m_VertexCount = 0;
		uint32_t m_IndexCount = 0;
		AABB m_Bounds;

		Topology m_Topology = Topology::TriangleList;

		uint32_t m_UpdateCount = 0;

		friend struct GfxDrawingOperation;
		friend class TangentGenerator;
	};
}