#include "GfxDrawingOperation.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Mesh.h"
#include "..\Graphics\DefaultMaterials.h"

namespace Blueberry
{
	GfxDrawingOperation::GfxDrawingOperation(GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer, Material* material, VertexLayout* layout, const uint32_t& indexCount, const uint32_t& indexOffset, const uint32_t& vertexCount, const Topology& topology, const uint8_t& passIndex, GfxBuffer* instanceBuffer, const uint32_t& instanceOffset, const uint32_t& instanceCount)
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

		this->vertexBuffer = vertexBuffer;
		this->indexBuffer = indexBuffer;
		this->indexCount = indexCount;
		this->indexOffset = indexOffset;
		this->vertexCount = vertexCount;
		this->topology = topology;
		this->instanceBuffer = instanceBuffer;
		this->instanceOffset = instanceOffset;
		this->instanceCount = instanceCount;
		this->layout = layout;
		this->material = material;
		this->passIndex = passIndex;
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const uint32_t& indexCount, const uint32_t& indexOffset, const uint32_t& vertexCount, const uint8_t& passIndex, GfxBuffer* instanceBuffer, const uint32_t& instanceOffset, const uint32_t& instanceCount) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, &mesh->m_Layout, indexCount, indexOffset, vertexCount, mesh->GetTopology(), passIndex, instanceBuffer, instanceOffset, instanceCount)
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const uint8_t& passIndex, GfxBuffer* instanceBuffer, const uint32_t& instanceOffset, const uint32_t& instanceCount) : GfxDrawingOperation(mesh != nullptr && mesh->GetState() != ObjectState::Missing ? mesh->m_VertexBuffer : nullptr, mesh->m_IndexBuffer, material, &mesh->m_Layout, mesh->m_IndexCount, 0, mesh->m_VertexCount, mesh->GetTopology(), passIndex, instanceBuffer, instanceOffset, instanceCount)
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
}
