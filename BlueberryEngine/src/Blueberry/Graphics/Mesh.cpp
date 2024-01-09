#include "bbpch.h"
#include "Mesh.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Mesh)

	const UINT& Mesh::GetVertexCount()
	{
		return m_VertexCount;
	}

	const UINT& Mesh::GetIndexCount()
	{
		return m_IndexCount;
	}

	void Mesh::SetVertexData(float* data, const UINT& vertexCount)
	{
		m_VertexBuffer->SetData(data, vertexCount);
	}

	void Mesh::SetIndexData(UINT* data, const UINT& indexCount)
	{
		m_IndexBuffer->SetData(data, indexCount);
	}

	const Topology& Mesh::GetTopology()
	{
		return m_Topology;
	}

	void Mesh::SetTopology(const Topology& topology)
	{
		m_Topology = topology;
	}

	Mesh* Mesh::Create(const VertexLayout& layout, const UINT& vertexCount, const UINT& indexCount)
	{
		Mesh* mesh = Object::Create<Mesh>();
		GfxDevice::CreateVertexBuffer(layout, vertexCount, mesh->m_VertexBuffer);
		GfxDevice::CreateIndexBuffer(indexCount, mesh->m_IndexBuffer);
		mesh->m_VertexCount = vertexCount;
		mesh->m_IndexCount = indexCount;
		return mesh;
	}

	void Mesh::BindProperties()
	{
	}
}