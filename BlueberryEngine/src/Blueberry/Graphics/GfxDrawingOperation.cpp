#include "bbpch.h"
#include "GfxDrawingOperation.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Mesh.h"

namespace Blueberry
{
	GfxDrawingOperation::GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const Topology& topology)
	{
		shader = material->m_Shader->m_Shader.get();
		textures = &(material->m_GfxTextures);
		this->vertexBuffer = vertexBuffer;
		this->indexBuffer = indexBuffer;
		this->indexCount = indexCount;
		this->topology = topology;
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount) : GfxDrawingOperation(mesh->m_VertexBuffer.get(), mesh->m_IndexBuffer.get(), material, indexCount, mesh->GetTopology())
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material) : GfxDrawingOperation(mesh->m_VertexBuffer.get(), mesh->m_IndexBuffer.get(), material, mesh->m_IndexCount, mesh->GetTopology())
	{
	}

	bool GfxDrawingOperation::IsValid() const
	{
		return shader != nullptr && vertexBuffer != nullptr && indexBuffer != nullptr && indexCount > 0 && topology != Topology::Unknown;
	}
}
