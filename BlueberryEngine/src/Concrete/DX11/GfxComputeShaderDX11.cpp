#include "GfxComputeShaderDX11.h"

#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	GfxComputeShaderDX11::GfxComputeShaderDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	bool GfxComputeShaderDX11::Initialize(void* computeData)
	{
		if (computeData == nullptr)
		{
			BB_ERROR("Compute data is empty.");
			return false;
		}

		m_ComputeShaderBuffer.Attach(static_cast<ID3DBlob*>(computeData));
		HRESULT hr = m_Device->CreateComputeShader(m_ComputeShaderBuffer->GetBufferPointer(), m_ComputeShaderBuffer->GetBufferSize(), NULL, m_ComputeShader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create compute shader from data.");
			return false;
		}

		ID3D11ShaderReflection* computeShaderReflection;
		hr = D3DReflect(m_ComputeShaderBuffer->GetBufferPointer(), m_ComputeShaderBuffer->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&computeShaderReflection);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get compute shader reflection."));
			return false;
		}

		D3D11_SHADER_DESC computeShaderDesc;
		computeShaderReflection->GetDesc(&computeShaderDesc);

		unsigned int resourceBindingCount = computeShaderDesc.BoundResources;

		m_SRVSlots.reserve(16);
		m_ConstantBufferSlots.reserve(14);
		m_UAVSlots.reserve(8);
		m_SamplerSlots.reserve(16);

		for (uint32_t i = 0; i < resourceBindingCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			computeShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			switch (inputBindDesc.Type)
			{
			case D3D_SIT_TEXTURE:
				m_SRVSlots.insert(m_SRVSlots.begin() + inputBindDesc.BindPoint, TO_HASH(String(inputBindDesc.Name)));
				break;
			case D3D_SIT_CBUFFER:
				m_ConstantBufferSlots.insert(m_ConstantBufferSlots.begin() + inputBindDesc.BindPoint, TO_HASH(String(inputBindDesc.Name)));
				break;
			case D3D_SIT_STRUCTURED:
				m_SRVSlots.insert(m_SRVSlots.begin() + inputBindDesc.BindPoint, TO_HASH(String(inputBindDesc.Name)));
				break;
			case D3D_SIT_UAV_RWTYPED:
				m_UAVSlots.insert(m_UAVSlots.begin() + inputBindDesc.BindPoint, TO_HASH(String(inputBindDesc.Name)));
				break;
			case D3D10_SIT_SAMPLER:
			{
				String samplerName = String(inputBindDesc.Name);
				auto pos = samplerName.find("_Sampler");
				if (pos != std::string::npos)
				{
					samplerName.replace(pos, samplerName.length() - pos, "");
				}
				else
				{
					BB_ERROR("Wrong sampler name.");
				}
				m_SamplerSlots.insert(m_SamplerSlots.begin() + inputBindDesc.BindPoint, TO_HASH(samplerName));
			}
			break;
			default:
				BB_ERROR("Missing input type.");
				break;
			}
		}
		return true;
	}
}