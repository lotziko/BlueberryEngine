#include "bbpch.h"
#include "GfxDrawingOperation.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\DefaultMaterials.h"

namespace Blueberry
{
	GfxDrawingOperation::GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const UINT& indexOffset, const UINT& vertexCount, const Topology& topology)
	{
		// TODO move into pipeline state
		if (vertexBuffer == nullptr || topology == Topology::Unknown)
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

		this->renderState = material->GetState(0);
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
		this->materialId = material->GetObjectId();
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset, const UINT& vertexCount) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, indexCount, indexOffset, vertexCount, mesh->GetTopology())
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material) : GfxDrawingOperation(mesh != nullptr && mesh->GetState() != ObjectState::Missing ? mesh->m_VertexBuffer : nullptr, mesh->m_IndexBuffer, material, mesh->m_IndexCount, 0, mesh->m_VertexCount, mesh->GetTopology())
	{
	}

	bool GfxDrawingOperation::IsValid() const
	{
		return isValid;
	}
}
