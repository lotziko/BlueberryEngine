#include "bbpch.h"
#include "GfxDrawingOperation.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Mesh.h"

namespace Blueberry
{
	GfxDrawingOperation::GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount)
	{
		shader = material->m_Shader->m_Shader.get();
		textures = &(material->m_GfxTextures);
		vertexBuffer = mesh->m_VertexBuffer.get();
		indexBuffer = mesh->m_IndexBuffer.get();
		this->indexCount = indexCount;
	}
}
