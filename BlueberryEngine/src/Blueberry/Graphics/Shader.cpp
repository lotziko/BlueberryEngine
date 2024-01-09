#include "bbpch.h"
#include "Shader.h"

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Shader)

	void Shader::Initialize(void* vertexData, void* pixelData)
	{
		GfxDevice::CreateShader(vertexData, pixelData, m_Shader);
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