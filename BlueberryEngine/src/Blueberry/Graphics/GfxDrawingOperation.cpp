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
		shader = (material->m_Shader.IsValid() && material->m_Shader->IsValid()) ? material->m_Shader->m_Shader : nullptr;
		textures = &(material->m_GfxTextures);
		this->vertexBuffer = vertexBuffer;
		this->indexBuffer = indexBuffer;
		this->indexCount = indexCount;
		this->indexOffset = indexOffset;
		this->cullMode = material->m_CullMode;
		this->surfaceType = material->m_SurfaceType;
		this->topology = topology;
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, indexCount, indexOffset, mesh->GetTopology())
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material)
	{
		shader = (material != nullptr && material->IsValid() && material->m_Shader.IsValid() && material->m_Shader->IsValid()) ? material->m_Shader->m_Shader : nullptr;
		if (mesh != nullptr && mesh->IsValid() && mesh->GetVertexCount() > 0)
		{
			textures = &(material->m_GfxTextures);
			vertexBuffer = mesh->m_VertexBuffer;
			indexBuffer = mesh->m_IndexBuffer;
			indexCount = mesh->m_IndexCount;
			indexOffset = 0;
			cullMode = material->m_CullMode;
			surfaceType = material->m_SurfaceType;
			topology = mesh->GetTopology();
		}
		else
		{
			textures = {};
			vertexBuffer = nullptr;
			indexBuffer = nullptr;
			indexCount = 0;
			indexOffset = 0;
			cullMode = CullMode::None;
			surfaceType = SurfaceType::Opaque;
			topology = Topology::Unknown;
		}
	}

	bool GfxDrawingOperation::IsValid() const
	{
		return shader != nullptr && vertexBuffer != nullptr && indexBuffer != nullptr && indexCount > 0 && topology != Topology::Unknown;
	}
}
