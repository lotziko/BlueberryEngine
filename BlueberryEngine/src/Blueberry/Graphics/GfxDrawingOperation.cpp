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
		// TODO move into pipeline state
		if (vertexBuffer == nullptr || indexBuffer == nullptr || indexCount == 0 || topology == Topology::Unknown)
		{
			isValid = false;
			return;
		}

		if (material == nullptr || !material->IsDefaultState() || !material->m_Shader.IsValid() || !material->m_Shader->IsDefaultState())
		{
			material = DefaultMaterials::GetError();
		}

		if (!material->m_Shader.IsValid() || !material->m_Shader->IsDefaultState())
		{
			isValid = false;
			return;
		}

		auto flags = material->GetKeywordFlags();
		auto variant = material->m_Shader->GetVariant(flags.first, flags.second);
		this->vertexShader = variant.vertexShader;
		this->geometryShader = variant.geometryShader;
		this->fragmentShader = variant.fragmentShader;
		this->textureCount = 0;

		for (auto& pair : material->m_TextureMap)
		{
			if (pair.second.IsValid())
			{
				textures[textureCount] = std::make_pair(pair.first, pair.second.Get()->m_Texture);
				++textureCount;
			}
		}

		auto& data = *(material->GetShaderData());
		auto pass = data.GetPass(0);
		if (pass == nullptr)
		{
			isValid = false;
			return;
		}
		this->cullMode = pass->GetCullMode();
		this->blendSrc = pass->GetBlendSrc();
		this->blendDst = pass->GetBlendDst();
		this->zWrite = pass->GetZWrite();

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
