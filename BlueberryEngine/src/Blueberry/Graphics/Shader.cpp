#include "bbpch.h"
#include "Shader.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Shader)

	Shader::Shader(std::wstring shaderPath)
	{
		g_GraphicsDevice->CreateShader(shaderPath, m_Shader);
	}

	Ref<Shader> Shader::Create(std::wstring shaderPath)
	{
		return ObjectDB::CreateObject<Shader>(shaderPath);
	}

	void Shader::BindProperties()
	{
	}
}