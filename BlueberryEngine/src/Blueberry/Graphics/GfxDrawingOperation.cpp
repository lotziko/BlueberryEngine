#include "bbpch.h"
#include "GfxDrawingOperation.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\DefaultMaterials.h"

namespace Blueberry
{
	GfxDrawingOperation::GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const UINT& indexOffset, const UINT& vertexCount, const Topology& topology, const uint8_t& passIndex, GfxVertexBuffer* instanceBuffer, const UINT& instanceOffset)
	{
		// TODO move into pipeline state
		if (vertexBuffer == nullptr || topology == Topology::Unknown)
		{
			isValid = false;
			return;
		}

		if (!material->m_Shader.IsValid() || !material->m_Shader->IsDefaultState())
		{
			isValid = false;
			return;
		}

		this->renderState = material->GetState(passIndex);
		if (this->renderState == nullptr)
		{
			isValid = false;
			return;
		}

		this->vertexBuffer = vertexBuffer;
		this->indexBuffer = indexBuffer;
		this->indexCount = indexCount;
		this->indexOffset = indexOffset;
		this->vertexCount = vertexCount;
		this->topology = topology;
		this->instanceBuffer = instanceBuffer;
		this->instanceOffset = instanceOffset;
		this->materialId = material->GetObjectId();
		this->materialCRC = material->GetCRC();
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset, const UINT& vertexCount, const uint8_t& passIndex, GfxVertexBuffer* instanceBuffer, const UINT& instanceOffset) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, indexCount, indexOffset, vertexCount, mesh->GetTopology(), passIndex, instanceBuffer, instanceOffset)
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const uint8_t& passIndex, GfxVertexBuffer* instanceBuffer, const UINT& instanceOffset) : GfxDrawingOperation(mesh != nullptr && mesh->GetState() != ObjectState::Missing ? mesh->m_VertexBuffer : nullptr, mesh->m_IndexBuffer, material, mesh->m_IndexCount, 0, mesh->m_VertexCount, mesh->GetTopology(), passIndex, instanceBuffer, instanceOffset)
	{
	}

	Material* GfxDrawingOperation::GetValidMaterial(Material* material)
	{
		if (material == nullptr || !material->IsDefaultState() || !material->m_Shader.IsValid() || !material->m_Shader->IsDefaultState())
		{
			material = DefaultMaterials::GetError();
		}
		return material;
	}

	bool GfxDrawingOperation::IsValid() const
	{
		return isValid;
	}
	
	GfxTexture* GfxRenderState::TextureInfo::Get()
	{
		Texture* texture = static_cast<Texture*>(ObjectDB::GetObject(*textureId));
		return texture == nullptr ? nullptr : texture->m_Texture;
	}
}
