#include "Blueberry\Graphics\Mesh.h"

#include "Blueberry\Core\ClassDB.h"
#include "..\Graphics\GfxDevice.h"
#include "..\Graphics\GfxBuffer.h"
#include "Blueberry\Tools\CRCHelper.h"

#include <mikktspace\mikktspace.h>

namespace Blueberry
{
	DATA_DEFINITION(SubMeshData)
	{
		DEFINE_FIELD(SubMeshData, m_IndexStart, BindingType::Int, {})
		DEFINE_FIELD(SubMeshData, m_IndexCount, BindingType::Int, {})
	}

	OBJECT_DEFINITION(Mesh, Object)
	{
		DEFINE_BASE_FIELDS(Mesh, Object)
		DEFINE_FIELD(Mesh, m_VertexData, BindingType::FloatList, {})
		DEFINE_FIELD(Mesh, m_IndexData, BindingType::IntList, {})
		DEFINE_FIELD(Mesh, m_VertexCount, BindingType::Int, {})
		DEFINE_FIELD(Mesh, m_IndexCount, BindingType::Int, {})
		DEFINE_FIELD(Mesh, m_SubMeshes, BindingType::DataList, FieldOptions().SetObjectType(SubMeshData::Type))
		DEFINE_FIELD(Mesh, m_Layout, BindingType::Raw, FieldOptions().SetSize(sizeof(VertexLayout)))
		DEFINE_FIELD(Mesh, m_Bounds, BindingType::AABB, {})
	}

	const uint32_t& SubMeshData::GetIndexStart() const
	{
		return m_IndexStart;
	}

	void SubMeshData::SetIndexStart(const uint32_t& indexStart)
	{
		m_IndexStart = indexStart;
	}

	const uint32_t& SubMeshData::GetIndexCount() const
	{
		return m_IndexCount;
	}

	void SubMeshData::SetIndexCount(const uint32_t& indexCount)
	{
		m_IndexCount = indexCount;
	}

	const int VERTICES_BIT = 1;
	const int NORMALS_BIT = 2;
	const int TANGENTS_BIT = 4;
	const int UV0_BIT = 8;

	Mesh::~Mesh()
	{
		if (m_VertexBuffer != nullptr)
		{
			delete m_VertexBuffer;
		}
		if (m_IndexBuffer != nullptr)
		{
			delete m_IndexBuffer;
		}
	}

	const uint32_t& Mesh::GetVertexCount()
	{
		return m_VertexCount;
	}

	const uint32_t& Mesh::GetIndexCount()
	{
		return m_IndexCount;
	}

	const uint32_t Mesh::GetSubMeshCount()
	{
		return static_cast<uint32_t>(m_SubMeshes.size());
	}

	const SubMeshData& Mesh::GetSubMesh(const uint32_t& index)
	{
		return m_SubMeshes[index];
	}

	const List<Vector3>& Mesh::GetVertices()
	{
		if (m_Vertices.size() == 0 && m_VertexData.size() > 0)
		{
			m_Vertices.reserve(m_VertexCount);
			float* begin = m_VertexData.data();
			float* end = begin + m_VertexData.size();
			uint32_t vertexSize = static_cast<uint32_t>(m_VertexData.size() / m_VertexCount);
			for (float* it = begin; it < end; it += vertexSize)
			{
				m_Vertices.emplace_back(Vector3(*it, *(it + 1), *(it + 2)));
			}
		}

		return m_Vertices;
	}

	const List<uint32_t>& Mesh::GetIndices()
	{
		if (m_Indices.size() == 0)
		{
			return m_IndexData;
		}

		return m_Indices;
	}

	void Mesh::SetVertices(const Vector3* vertices, const uint32_t& vertexCount)
	{
		m_VertexCount = vertexCount;
		m_Vertices.resize(vertexCount);
		memcpy(m_Vertices.data(), vertices, sizeof(Vector3) * vertexCount);
		m_BufferIsDirty = true;
	}

	void Mesh::SetNormals(const Vector3* normals, const uint32_t& vertexCount)
	{
		m_Normals.resize(vertexCount);
		memcpy(m_Normals.data(), normals, sizeof(Vector3) * vertexCount);
		m_BufferIsDirty = true;
	}

	void Mesh::SetTangents(const Vector4* tangents, const uint32_t& vertexCount)
	{
		m_Tangents.resize(vertexCount);
		memcpy(m_Tangents.data(), tangents, sizeof(Vector4) * vertexCount);
		m_BufferIsDirty = true;
	}

	void Mesh::SetColors(const Color* colors, const uint32_t& vertexCount)
	{
		m_Colors.resize(vertexCount);
		memcpy(m_Colors.data(), colors, sizeof(Color) * vertexCount);
		m_BufferIsDirty = true;
	}

	void Mesh::SetIndices(const uint32_t* indices, const uint32_t& indexCount)
	{
		m_IndexCount = indexCount;
		m_Indices.resize(indexCount);
		memcpy(m_Indices.data(), indices, sizeof(uint32_t) * indexCount);
		m_BufferIsDirty = true;
	}

	void Mesh::SetUVs(const int& channel, const Vector2* uvs, const uint32_t& uvCount)
	{
		if (channel < 0 || channel >= 8)
		{
			return;
		}
		if (uvCount == m_VertexCount)
		{
			m_UVs[channel].resize(uvCount);
			memcpy(m_UVs[channel].data(), uvs, sizeof(Vector2) * uvCount);
			m_BufferIsDirty = true;
		}
	}

	void Mesh::SetSubMesh(const uint32_t& index, const SubMeshData& data)
	{
		if (index >= m_SubMeshes.size())
		{
			m_SubMeshes.resize(index + 1);
		}
		m_SubMeshes[index] = data;
	}

	// Based on https://www.turais.de/using-mikktspace-in-your-project/
	class TangentGenerator
	{
	public:
		void Generate(Mesh* mesh)
		{
			m_Iface.m_getNumFaces = GetNumFaces;
			m_Iface.m_getNumVerticesOfFace = GetNumVerticesOfFace;

			m_Iface.m_getNormal = GetNormal;
			m_Iface.m_getPosition = GetPosition;
			m_Iface.m_getTexCoord = GetTexCoords;
			m_Iface.m_setTSpaceBasic = SetTspaceBasic;

			m_Context.m_pInterface = &m_Iface;
			m_Context.m_pUserData = mesh;

			genTangSpaceDefault(&this->m_Context);
		}

	private:
		static int GetNumFaces(const SMikkTSpaceContext* context)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			return mesh->m_IndexCount / 3;
		}

		static int GetNumVerticesOfFace(const SMikkTSpaceContext* context, int iFace)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			return 3;
		}

		static void GetPosition(const SMikkTSpaceContext* context, float outpos[], int iFace, int iVert)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			Vector3 position = mesh->m_Vertices[mesh->m_Indices[iFace * 3 + iVert]];

			outpos[0] = position.x;
			outpos[1] = position.y;
			outpos[2] = position.z;
		}

		static void GetNormal(const SMikkTSpaceContext* context, float outnormal[], int iFace, int iVert)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			Vector3 normal = mesh->m_Normals[mesh->m_Indices[iFace * 3 + iVert]];

			outnormal[0] = normal.x;
			outnormal[1] = normal.y;
			outnormal[2] = normal.z;
		}

		static void GetTexCoords(const SMikkTSpaceContext* context, float outuv[], int iFace, int iVert)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			Vector2 uv = mesh->m_UVs[0][mesh->m_Indices[iFace * 3 + iVert]];

			outuv[0] = uv.x;
			outuv[1] = uv.y;
		}

		static void SetTspaceBasic(const SMikkTSpaceContext *context, const float tangentu[], float fSign, int iFace, int iVert)
		{
			Vector4 tangent;
			tangent.x = tangentu[0];
			tangent.y = tangentu[1];
			tangent.z = tangentu[2];
			tangent.w = fSign;

			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			mesh->m_Tangents[mesh->m_Indices[iFace * 3 + iVert]] = tangent;
		}

	private:
		SMikkTSpaceInterface m_Iface = {};
		SMikkTSpaceContext m_Context = {};
	};
	
	void Mesh::GenerateTangents()
	{
		m_Tangents.resize(m_VertexCount);
		TangentGenerator generator;
		generator.Generate(this);
		m_BufferIsDirty = true;
	}

	const Topology& Mesh::GetTopology()
	{
		return m_Topology;
	}

	void Mesh::SetTopology(const Topology& topology)
	{
		m_Topology = topology;
	}

	const AABB& Mesh::GetBounds()
	{
		return m_Bounds;
	}

	void Mesh::Apply()
	{
		if (m_BufferIsDirty)
		{
			m_Layout = {};
			size_t vertexBufferSize = 0;
			uint32_t offset = 0;
			if (m_VertexCount > 0)
			{
				vertexBufferSize += m_VertexCount * sizeof(Vector3) / sizeof(float);
				m_Layout.Append(VertexAttribute::Position, 12);
			}
			if (m_Normals.size() == m_VertexCount)
			{
				vertexBufferSize += m_VertexCount * sizeof(Vector3) / sizeof(float);
				m_Layout.Append(VertexAttribute::Normal, 12);
			}
			if (m_Tangents.size() == m_VertexCount)
			{
				vertexBufferSize += m_VertexCount * sizeof(Vector4) / sizeof(float);
				m_Layout.Append(VertexAttribute::Tangent, 16);
			}
			for (int i = 0; i < 4; ++i)
			{
				if (m_UVs[i].size() == m_VertexCount)
				{
					vertexBufferSize += m_VertexCount * sizeof(Vector2) / sizeof(float);
					m_Layout.Append(static_cast<VertexAttribute>(static_cast<uint32_t>(VertexAttribute::Texcoord0) + i), 8);
				}
			}
			m_Layout.Apply();

			if (vertexBufferSize != m_VertexData.size())
			{
				m_VertexData.resize(vertexBufferSize);
			}

			float* bufferPointer = m_VertexData.data();
			Vector3* vertexPointer = m_Vertices.data();
			Vector3* normalPointer = m_Normals.data();
			Vector4* tangentPointer = m_Tangents.data();
			Vector2* uvPointer = m_UVs[0].data();

			for (uint32_t i = 0; i < m_VertexCount; ++i)
			{
				memcpy(bufferPointer, vertexPointer, sizeof(Vector3));
				bufferPointer += 3;
				vertexPointer += 1;
				if (normalPointer != nullptr)
				{
					memcpy(bufferPointer, normalPointer, sizeof(Vector3));
					bufferPointer += 3;
					normalPointer += 1;
				}
				if (tangentPointer != nullptr)
				{
					memcpy(bufferPointer, tangentPointer, sizeof(Vector4));
					bufferPointer += 4;
					tangentPointer += 1;
				}
				if (uvPointer != nullptr)
				{
					memcpy(bufferPointer, uvPointer, sizeof(Vector2));
					bufferPointer += 2;
					uvPointer += 1;
				}
			}

			if (m_IndexCount != m_IndexData.size())
			{
				m_IndexData.resize(m_IndexCount);
			}
			memcpy(m_IndexData.data(), m_Indices.data(), m_IndexCount * sizeof(uint32_t));

			AABB::CreateFromPoints(m_Bounds, m_VertexCount, m_Vertices.data(), sizeof(Vector3));
		}

		// TODO handle old buffers instead
		GfxDevice::CreateVertexBuffer(m_VertexCount, static_cast<uint32_t>((m_VertexData.size() * sizeof(float)) / m_VertexCount), m_VertexBuffer);
		GfxDevice::CreateIndexBuffer(m_IndexCount, m_IndexBuffer);

		m_VertexBuffer->SetData(m_VertexData.data(), m_VertexCount);
		m_IndexBuffer->SetData(m_IndexData.data(), m_IndexCount);
	}

	const VertexLayout& Mesh::GetLayout()
	{
		return m_Layout;
	}

	Mesh* Mesh::Create()
	{
		Mesh* mesh = Object::Create<Mesh>();
		return mesh;
	}
}