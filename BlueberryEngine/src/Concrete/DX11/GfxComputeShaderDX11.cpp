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

		uint32_t constantBufferCount = computeShaderDesc.ConstantBuffers;

		for (uint32_t i = 0; i < constantBufferCount; i++)
		{
			ID3D11ShaderReflectionConstantBuffer* constantBufferReflection = computeShaderReflection->GetConstantBufferByIndex(i);
			D3D11_SHADER_BUFFER_DESC shaderBufferDesc;
			constantBufferReflection->GetDesc(&shaderBufferDesc);
			if (shaderBufferDesc.Type == D3D_CT_CBUFFER)
			{
				m_ConstantBufferSlots.insert({ TO_HASH(String(shaderBufferDesc.Name)), i });
			}
			else if (shaderBufferDesc.Type == D3D_CT_RESOURCE_BIND_INFO)
			{
				m_StructuredBufferSlots.insert({ TO_HASH(String(shaderBufferDesc.Name)), std::make_pair(i, 0) });
			}
		}

		unsigned int resourceBindingCount = computeShaderDesc.BoundResources;

		for (uint32_t i = 0; i < resourceBindingCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			computeShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			if (inputBindDesc.Type == D3D_SIT_TEXTURE)
			{
				uint32_t samplerSlot = -1;
				if (inputBindDesc.NumSamples > 8)
				{
					for (uint32_t j = 0; j < resourceBindingCount; j++)
					{
						D3D11_SHADER_INPUT_BIND_DESC samplerInputBindDesc;
						computeShaderReflection->GetResourceBindingDesc(j, &samplerInputBindDesc);
						if (samplerInputBindDesc.Type == D3D10_SIT_SAMPLER)
						{
							if (strncmp(samplerInputBindDesc.Name, inputBindDesc.Name, strlen(inputBindDesc.Name)) == 0)
							{
								samplerSlot = samplerInputBindDesc.BindPoint;
								break;
							}
						}
					}
				}
				m_TextureSlots.insert({ TO_HASH(String(inputBindDesc.Name)), std::make_pair(inputBindDesc.BindPoint, samplerSlot) });
			}
			else if (inputBindDesc.Type == D3D_SIT_UAV_RWTYPED)
			{
				m_ComputeBufferSlots.insert({ TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint });
			}
			else if (inputBindDesc.Type == D3D_SIT_STRUCTURED)
			{
				auto pair = &m_StructuredBufferSlots[TO_HASH(String(inputBindDesc.Name))];
				pair->second = i;
			}
		}

		return true;
	}
}