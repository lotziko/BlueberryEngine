#include "GfxComputeShaderDX11.h"

#include "..\Windows\WindowsHelper.h"
#include "..\..\Blueberry\Graphics\SamplerHelper.h"

namespace Blueberry
{
	bool GfxComputeShaderDX11::Initialize(ID3D11Device* device, const ByteData& computeData)
	{
		if (computeData.size() == 0)
		{
			BB_ERROR("Compute data is empty.");
			return false;
		}

		HRESULT hr = device->CreateComputeShader(computeData.data(), computeData.size(), NULL, m_ComputeShader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create compute shader from data.");
			return false;
		}

		ComPtr<ID3D11ShaderReflection> computeShaderReflection;
		hr = D3DReflect(computeData.data(), computeData.size(), IID_ID3D11ShaderReflection, (void**)computeShaderReflection.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get compute shader reflection."));
			return false;
		}

		D3D11_SHADER_DESC computeShaderDesc;
		computeShaderReflection->GetDesc(&computeShaderDesc);

		unsigned int resourceBindingCount = computeShaderDesc.BoundResources;

		for (uint32_t i = 0; i < resourceBindingCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			computeShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			switch (inputBindDesc.Type)
			{
			case D3D_SIT_TEXTURE:
				m_TextureSRVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
				break;
			case D3D_SIT_CBUFFER:
				m_ConstantBufferSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
				break;
			case D3D_SIT_STRUCTURED:
				m_BufferSRVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
				break;
			case D3D_SIT_BYTEADDRESS:
				m_BufferSRVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
				break;
			case D3D_SIT_UAV_RWBYTEADDRESS:
				m_BufferUAVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
				break;
			case D3D_SIT_UAV_RWTYPED:
				if (inputBindDesc.Dimension == D3D_SRV_DIMENSION_BUFFER)
				{
					m_BufferUAVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
				}
				else
				{
					m_TextureUAVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
				}
				break;
			case D3D_SIT_SAMPLER:
			{
				String samplerName = String(inputBindDesc.Name);
				size_t samplerHash = TO_HASH(samplerName);

				FilterMode filterMode;
				WrapMode wrapMode;
				if (SamplerHelper::ParseName(samplerHash, filterMode, wrapMode))
				{
					m_StaticSamplerSlots.push_back(std::make_tuple(filterMode, wrapMode, inputBindDesc.BindPoint));
				}
				else
				{
					auto pos = samplerName.find("_Sampler");
					if (pos != std::string::npos)
					{
						samplerName.replace(pos, samplerName.length() - pos, "");
						m_SamplerSlots.push_back(std::make_pair(TO_HASH(samplerName), inputBindDesc.BindPoint));
					}
					else
					{
						BB_ERROR("Wrong sampler name.");
					}
				}
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