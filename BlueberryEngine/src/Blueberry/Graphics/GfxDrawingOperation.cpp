#include "bbpch.h"
#include "GfxDrawingOperation.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\DefaultMaterials.h"

namespace Blueberry
{
	GfxDrawingOperation::GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const UINT& indexOffset, const Topology& topology)
	{
		if (vertexBuffer == nullptr || indexBuffer == nullptr || indexCount == 0 || topology == Topology::Unknown)
		{
			isValid = false;
			return;
		}

		if (material == nullptr || !material->IsDefaultState() || !material->m_Shader.IsValid() || !material->m_Shader->IsDefaultState())
		{
			material = DefaultMaterials::GetError();
		}
		shader = (material->m_Shader.IsValid() && material->m_Shader->IsDefaultState()) ? material->m_Shader->m_Shader : nullptr;
		if (shader == nullptr)
		{
			isValid = false;
			return;
		}

		for (auto& pair : material->m_TextureMap)
		{
			auto slot = shader->m_TextureSlots.find(pair.first);
			if (slot != shader->m_TextureSlots.end())
			{
				textures[slot->second] = pair.second.Get()->m_Texture;
			}
		}

		auto& data = *(material->GetShaderData());
		this->cullMode = data.GetCullMode();
		this->blendSrc = data.GetBlendSrc();
		this->blendDst = data.GetBlendDst();
		this->zWrite = data.GetZWrite();

		this->vertexBuffer = vertexBuffer;
		this->indexBuffer = indexBuffer;
		this->indexCount = indexCount;
		this->indexOffset = indexOffset;
		this->topology = topology;
		this->materialId = material->GetObjectId();
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, indexCount, indexOffset, mesh->GetTopology())
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material) : GfxDrawingOperation(mesh != nullptr && mesh->GetState() != ObjectState::Missing ? mesh->m_VertexBuffer : nullptr, mesh->m_IndexBuffer, material, mesh->m_IndexCount, 0, mesh->GetTopology())
	{
	}

	bool GfxDrawingOperation::IsValid() const
	{
		return isValid;
	}
}
