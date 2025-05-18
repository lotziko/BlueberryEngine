#include "ComputeShader.h"

#include "..\Graphics\GfxDevice.h"
#include "..\Graphics\GfxComputeShader.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ComputeShader, Object)
	{
	}

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
}