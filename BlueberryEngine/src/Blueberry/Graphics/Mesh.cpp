#include "bbpch.h"
#include "Mesh.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

#include "mikktspace\mikktspace.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Mesh)

	const int VERTICES_BIT = 1;
	const int NORMALS_BIT = 2;
	const int TANGENTS_BIT = 4;
	const int UV0_BIT = 8;

	Mesh::~Mesh()
	{
		if (m_VertexBuffer != nullptr)
		{
			delete m_VertexBuffer;
			delete m_Vertices;
		}
		if (m_IndexBuffer != nullptr)
		{
			delete m_IndexBuffer;
			delete m_Indices;
		}
	}

	const UINT& Mesh::GetVertexCount()
	{
		return m_VertexCount;
	}

	const UINT& Mesh::GetIndexCount()
	{
		return m_IndexCount;
	}

	void Mesh::SetVertices(const Vector3* vertices, const UINT& vertexCount)
	{
		if (m_Vertices == nullptr || vertexCount > m_VertexCount)
		{
			m_VertexCount = vertexCount;
			m_Vertices = new Vector3[vertexCount];
			memcpy(m_Vertices, vertices, sizeof(Vector3) * vertexCount);
			m_ChannelFlags |= VERTICES_BIT;
			m_BufferIsDirty = true;
		}
	}

	void Mesh::SetNormals(const Vector3* normals, const UINT& vertexCount)
	{
		if (m_Normals == nullptr || vertexCount > m_VertexCount)
		{
			m_Normals = new Vector3[vertexCount];
			memcpy(m_Normals, normals, sizeof(Vector3) * vertexCount);
			m_ChannelFlags |= NORMALS_BIT;
			m_BufferIsDirty = true;
		}
	}

	void Mesh::SetTangents(const Vector4* tangents, const UINT& vertexCount)
	{
		if (m_Tangents == nullptr || vertexCount > m_VertexCount)
		{
			m_Tangents = new Vector4[vertexCount];
			memcpy(m_Tangents, tangents, sizeof(Vector4) * vertexCount);
			m_ChannelFlags |= TANGENTS_BIT;
			m_BufferIsDirty = true;
		}
	}

	void Mesh::SetIndices(const UINT* indices, const UINT& indexCount)
	{
		if (indexCount > m_IndexCount)
		{
			m_IndexCount = indexCount;
			m_Indices = new UINT[indexCount];
			memcpy(m_Indices, indices, sizeof(UINT) * indexCount);
			m_BufferIsDirty = true;
		}
	}

	void Mesh::SetUVs(const int& channel, const Vector2* uvs, const UINT& uvCount)
	{
		if (channel < 0 || channel >= 8)
		{
			return;
		}
		if (uvCount == m_VertexCount)
		{
			m_UVs[channel] = new Vector2[uvCount];
			memcpy(m_UVs[channel], uvs, sizeof(Vector2) * uvCount);
			m_ChannelFlags |= UV0_BIT;
			m_BufferIsDirty = true;
		}
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
		m_Tangents = new Vector4[m_VertexCount];
		TangentGenerator generator;
		generator.Generate(this);
		m_ChannelFlags |= TANGENTS_BIT;
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
			size_t vertexBufferSize = 0;
			if (m_VertexCount > 0)
			{
				vertexBufferSize += m_VertexCount * sizeof(Vector3) / sizeof(float);
			}
			if (m_Normals != nullptr)
			{
				vertexBufferSize += m_VertexCount * sizeof(Vector3) / sizeof(float);
			}
			if (m_Tangents != nullptr)
			{
				vertexBufferSize += m_VertexCount * sizeof(Vector4) / sizeof(float);
			}
			for (int i = 0; i < 8; ++i)
			{
				if (m_UVs[i] != nullptr)
				{
					vertexBufferSize += m_VertexCount * sizeof(Vector2) / sizeof(float);
				}
			}

			if (vertexBufferSize != m_VertexData.size())
			{
				m_VertexData.resize(vertexBufferSize);
			}

			float* bufferPointer = m_VertexData.data();
			Vector3* vertexPointer = m_Vertices;
			Vector3* normalPointer = m_Normals;
			Vector4* tangentPointer = m_Tangents;
			Vector2* uvPointer = m_UVs[0];

			for (UINT i = 0; i < m_VertexCount; ++i)
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
			memcpy(m_IndexData.data(), m_Indices, m_IndexCount * sizeof(UINT));

			AABB::CreateFromPoints(m_Bounds, m_VertexCount, m_Vertices, sizeof(Vector3));
		}

		// TODO handle old buffers instead
		GfxDevice::CreateVertexBuffer(GetLayout(), m_VertexCount, m_VertexBuffer);
		GfxDevice::CreateIndexBuffer(m_IndexCount, m_IndexBuffer);

		m_VertexBuffer->SetData(m_VertexData.data(), m_VertexCount);
		m_IndexBuffer->SetData(m_IndexData.data(), m_IndexCount);
	}

	Mesh* Mesh::Create()
	{
		Mesh* mesh = Object::Create<Mesh>();
		return mesh;
	}

	void Mesh::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Mesh)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &Mesh::m_Name, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_VertexData), &Mesh::m_VertexData, BindingType::FloatByteArray))
		BIND_FIELD(FieldInfo(TO_STRING(m_IndexData), &Mesh::m_IndexData, BindingType::IntByteArray))
		BIND_FIELD(FieldInfo(TO_STRING(m_VertexCount), &Mesh::m_VertexCount, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_IndexCount), &Mesh::m_IndexCount, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_ChannelFlags), &Mesh::m_ChannelFlags, BindingType::Int))
		BIND_FIELD(FieldInfo(TO_STRING(m_Bounds), &Mesh::m_Bounds, BindingType::AABB))
		END_OBJECT_BINDING()
	}

	VertexLayout Mesh::GetLayout()
	{
		VertexLayout layout = VertexLayout{};
		if (m_ChannelFlags & VERTICES_BIT)
		{
			layout.Append(VertexLayout::ElementType::Position3D);
		}
		if (m_ChannelFlags & NORMALS_BIT)
		{
			layout.Append(VertexLayout::ElementType::Normal);
		}
		if (m_ChannelFlags & TANGENTS_BIT)
		{
			layout.Append(VertexLayout::ElementType::Tangent);
		}
		if (m_ChannelFlags & UV0_BIT)
		{
			layout.Append(VertexLayout::ElementType::TextureCoord);
		}
		return layout;
	}
}