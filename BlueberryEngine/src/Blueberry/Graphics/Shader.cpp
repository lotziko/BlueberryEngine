#include "bbpch.h"
#include "Shader.h"

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Shader)

	const ShaderOptions& Shader::GetOptions()
	{
		return m_Options;
	}

	void Shader::Initialize(void* vertexData, void* pixelData)
	{
		GfxDevice::CreateShader(vertexData, pixelData, m_Shader);
	}

	void Shader::Initialize(void* vertexData, void* pixelData, const RawShaderOptions& options)
	{
		GfxDevice::CreateShader(vertexData, pixelData, m_Shader);
		m_Options = ShaderOptions(options);
	}

	Shader* Shader::Create(void* vertexData, void* pixelData)
	{
		Shader* shader = Object::Create<Shader>();
		shader->Initialize(vertexData, pixelData);
		return shader;
	}

	Shader* Shader::Create(void* vertexData, void* pixelData, const RawShaderOptions& options)
	{
		Shader* shader = Object::Create<Shader>();
		shader->Initialize(vertexData, pixelData, options);
		return shader;
	}

	void Shader::BindProperties()
	{
	}
}