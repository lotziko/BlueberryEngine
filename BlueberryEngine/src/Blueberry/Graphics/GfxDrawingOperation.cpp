#include "bbpch.h"
#include "GfxDrawingOperation.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Mesh.h"

namespace Blueberry
{
	GfxDrawingOperation::GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const UINT& indexOffset, const Topology& topology)
	{
		shader = material->m_Shader.IsValid() ? material->m_Shader->m_Shader : nullptr;
		textures = &(material->m_GfxTextures);
		this->vertexBuffer = vertexBuffer;
		this->indexBuffer = indexBuffer;
		this->indexCount = indexCount;
		this->indexOffset = indexOffset;
		this->topology = topology;
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, indexCount, indexOffset, mesh->GetTopology())
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, mesh->m_IndexCount, 0, mesh->GetTopology())
	{
	}

	bool GfxDrawingOperation::IsValid() const
	{
		return shader != nullptr && vertexBuffer != nullptr && indexBuffer != nullptr && indexCount > 0 && topology != Topology::Unknown;
	}
}
