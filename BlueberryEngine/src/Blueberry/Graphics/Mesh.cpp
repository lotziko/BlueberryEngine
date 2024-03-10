#include "bbpch.h"
#include "Mesh.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Mesh)

	Mesh::~Mesh()
	{
		if (m_VertexData != nullptr)
		{
			delete[] m_VertexData;
		}
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
		}
	}

	void Mesh::SetNormals(const Vector3* normals, const UINT& vertexCount)
	{
		if (m_Normals == nullptr || vertexCount > m_VertexCount)
		{
			m_Normals = new Vector3[vertexCount];
			memcpy(m_Normals, normals, sizeof(Vector3) * vertexCount);
		}
	}

	void Mesh::SetIndices(const UINT* indices, const UINT& indexCount)
	{
		if (indexCount > m_IndexCount)
		{
			m_IndexCount = indexCount;
			m_Indices = new UINT[indexCount];
			memcpy(m_Indices, indices, sizeof(UINT) * indexCount);
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
		VertexLayout layout = VertexLayout{};

		size_t vertexBufferSize = 0;
		if (m_VertexCount > 0)
		{
			vertexBufferSize += m_VertexCount * sizeof(Vector3) / sizeof(float);
			layout.Append(VertexLayout::ElementType::Position3D);
		}
		if (m_Normals != nullptr)
		{
			vertexBufferSize += m_VertexCount * sizeof(Vector3) / sizeof(float);
			layout.Append(VertexLayout::ElementType::Normal);
		}
		for (int i = 0; i < 8; ++i)
		{
			if (m_UVs[i] != nullptr)
			{
				vertexBufferSize += m_VertexCount * sizeof(Vector2) / sizeof(float);
				layout.Append(VertexLayout::ElementType::TextureCoord);
			}
		}

		if (m_VertexData == nullptr || m_VertexDataSize < vertexBufferSize)
		{
			m_VertexData = new float[vertexBufferSize];
			m_VertexDataSize = vertexBufferSize;
		}

		float* bufferPointer = m_VertexData;
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

		// TODO handle old buffers instead
		GfxDevice::CreateVertexBuffer(layout, m_VertexCount, m_VertexBuffer);
		GfxDevice::CreateIndexBuffer(m_IndexCount, m_IndexBuffer);

		m_VertexBuffer->SetData(m_VertexData, m_VertexCount);
		m_IndexBuffer->SetData(m_Indices, m_IndexCount);
	}

	Mesh* Mesh::Create()
	{
		Mesh* mesh = Object::Create<Mesh>();
		return mesh;
	}

	void Mesh::BindProperties()
	{
	}
}