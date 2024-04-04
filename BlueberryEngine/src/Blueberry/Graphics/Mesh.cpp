#include "bbpch.h"
#include "Mesh.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Mesh)

	const int VERTICES_BIT = 1;
	const int NORMALS_BIT = 2;
	const int UV0_BIT = 4;

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

	const Topology& Mesh::GetTopology()
	{
		return m_Topology;
	}

	void Mesh::SetTopology(const Topology& topology)
	{
		m_Topology = topology;
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
		if (m_ChannelFlags & UV0_BIT)
		{
			layout.Append(VertexLayout::ElementType::TextureCoord);
		}
		return layout;
	}
}