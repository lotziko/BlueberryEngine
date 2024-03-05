#include "bbpch.h"
#include "ComputeShader.h"

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, ComputeShader)

	void ComputeShader::Initialize(void* computeData)
	{
		GfxDevice::CreateComputeShader(computeData, m_Shader);
	}

	ComputeShader* ComputeShader::Create(void* computeData)
	{
		ComputeShader* shader = Object::Create<ComputeShader>();
		shader->Initialize(computeData);
		return shader;
	}

	void ComputeShader::BindProperties()
	{
	}
}