#include "bbpch.h"
#include "Mesh.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Graphics\GfxBuffer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Mesh)

	Mesh::Mesh(const VertexLayout& layout, const UINT& vertexCount, const UINT& indexCount)
	{
		g_GraphicsDevice->CreateVertexBuffer(layout, vertexCount, m_VertexBuffer);
		g_GraphicsDevice->CreateIndexBuffer(indexCount, m_IndexBuffer);
		m_VertexCount = vertexCount;
		m_IndexCount = indexCount;
	}

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

	Ref<Mesh> Mesh::Create(const VertexLayout& layout, const UINT& vertexCount, const UINT& indexCount)
	{
		return ObjectDB::CreateObject<Mesh>(layout, vertexCount, indexCount);
	}
}