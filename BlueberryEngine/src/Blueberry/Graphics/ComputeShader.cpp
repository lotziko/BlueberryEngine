#include "Blueberry\Graphics\ComputeShader.h"

#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "..\Graphics\GfxComputeShader.h"

namespace Blueberry
{
	DATA_DEFINITION(KernelData)
	{
		DEFINE_FIELD(KernelData, m_Name, BindingType::String, {})
	}

	DATA_DEFINITION(ComputeShaderData)
	{
		DEFINE_FIELD(ComputeShaderData, m_Kernels, BindingType::DataList, FieldOptions().SetObjectType(KernelData::Type))
	}

	OBJECT_DEFINITION(ComputeShader, Object)
	{
		DEFINE_BASE_FIELDS(ComputeShader, Object)
		DEFINE_FIELD(ComputeShader, m_Data, BindingType::Data, FieldOptions().SetObjectType(ComputeShaderData::Type))
	}

	void KernelData::SetName(const String& name)
	{
		m_Name = name;
	}

	const KernelData& ComputeShaderData::GetKernel(const uint32_t& index) const
	{
		return m_Kernels[index];
	}

	const size_t ComputeShaderData::GetKernelCount() const
	{
		return m_Kernels.size();
	}

	void ComputeShaderData::SetKernels(const List<KernelData>& kernels)
	{
		m_Kernels = kernels;
	}

	ComputeShader::~ComputeShader()
	{
		for (auto& shader : m_ComputeShaders)
		{
			delete shader;
		}
		m_ComputeShaders.clear();
	}

	void ComputeShader::Initialize(const List<void*>& shaders)
	{
		if (m_ComputeShaders.size() > 0)
		{
			for (auto it = m_ComputeShaders.begin(); it < m_ComputeShaders.end(); ++it)
			{
				delete *it;
			}
			m_ComputeShaders.clear();
		}

		size_t computeShadersCount = shaders.size();
		m_ComputeShaders.resize(computeShadersCount);
		for (size_t i = 0; i < computeShadersCount; ++i)
		{
			if (shaders[i] != nullptr)
			{
				GfxComputeShader* computeShader;
				GfxDevice::CreateComputeShader(shaders[i], computeShader);
				m_ComputeShaders[i] = computeShader;
			}
			else
			{
				m_ComputeShaders[i] = nullptr;
			}
		}
	}

	void ComputeShader::Initialize(const List<void*>& shaders, const ComputeShaderData& data)
	{
		Initialize(shaders);
		m_Data = data;
	}

	ComputeShader* ComputeShader::Create(const List<void*>& shaders, const ComputeShaderData& shaderData, ComputeShader* existingShader)
	{
		ComputeShader* shader = nullptr;
		if (existingShader != nullptr)
		{
			shader = existingShader;
		}
		else
		{
			shader = Object::Create<ComputeShader>();
		}
		shader->Initialize(shaders, shaderData);
		return shader;
	}

	GfxComputeShader* ComputeShader::GetKernel(const uint8_t& index)
	{
		if (index >= static_cast<uint8_t>(m_ComputeShaders.size()))
		{
			return nullptr;
		}
		return m_ComputeShaders[index];
	}
}