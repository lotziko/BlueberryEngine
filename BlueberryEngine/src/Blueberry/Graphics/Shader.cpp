#include "bbpch.h"
#include "Shader.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Shader)

	void Shader::Initialize(void* vertexData, void* pixelData)
	{
		g_GraphicsDevice->CreateShader(vertexData, pixelData, m_Shader);
	}

	Shader* Shader::Create(void* vertexData, void* pixelData)
	{
		Shader* shader = Object::Create<Shader>();
		shader->Initialize(vertexData, pixelData);
		return shader;
	}

	void Shader::BindProperties()
	{
	}
}