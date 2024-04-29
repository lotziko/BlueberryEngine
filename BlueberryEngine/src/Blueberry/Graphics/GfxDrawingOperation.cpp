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
		shader = (material->m_Shader.IsValid() && material->m_Shader->IsAlive()) ? material->m_Shader->m_Shader : nullptr;
		textures = &(material->m_GfxTextures);
		this->vertexBuffer = vertexBuffer;
		this->indexBuffer = indexBuffer;
		this->indexCount = indexCount;
		this->indexOffset = indexOffset;
		auto& options = material->GetShaderOptions();
		this->cullMode = options.GetCullMode();
		this->blendSrc = options.GetBlendSrc();
		this->blendDst = options.GetBlendDst();
		this->zWrite = options.GetZWrite();
		this->topology = topology;
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset) : GfxDrawingOperation(mesh->m_VertexBuffer, mesh->m_IndexBuffer, material, indexCount, indexOffset, mesh->GetTopology())
	{
	}

	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material)
	{
		if (material == nullptr || !material->IsAlive() || !material->m_Shader.IsValid() || !material->m_Shader->IsAlive())
		{
			material = DefaultMaterials::GetError();
		}
		
		shader = material->m_Shader->m_Shader;
		textures = &(material->m_GfxTextures);
		auto& options = material->GetShaderOptions();
		this->cullMode = options.GetCullMode();
		this->blendSrc = options.GetBlendSrc();
		this->blendDst = options.GetBlendDst();
		this->zWrite = options.GetZWrite();

		if (mesh != nullptr && mesh->IsAlive() && mesh->GetVertexCount() > 0)
		{
			vertexBuffer = mesh->m_VertexBuffer;
			indexBuffer = mesh->m_IndexBuffer;
			indexCount = mesh->m_IndexCount;
			indexOffset = 0;
			topology = mesh->GetTopology();
		}
		else
		{
			vertexBuffer = nullptr;
			indexBuffer = nullptr;
			indexCount = 0;
			indexOffset = 0;
			topology = Topology::Unknown;
		}
	}

	bool GfxDrawingOperation::IsValid() const
	{
		return shader != nullptr && vertexBuffer != nullptr && indexBuffer != nullptr && indexCount > 0 && topology != Topology::Unknown;
	}
}
