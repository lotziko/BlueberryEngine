#include "GfxShaderDX11.h"

#include "Blueberry\Tools\CRCHelper.h"
#include "..\..\Blueberry\Graphics\SamplerHelper.h"
#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	constexpr uint32_t String4ToInt(const char* s)
	{
		return (static_cast<uint32_t>(s[3]) << 24) | (static_cast<uint32_t>(s[2]) << 16) | (static_cast<uint32_t>(s[1]) << 8) |	static_cast<uint32_t>(s[0]);
	}

	bool GfxVertexShaderDX11::Initialize(ID3D11Device* device, const ByteData& vertexData)
	{
		if (vertexData.size() == 0)
		{
			BB_ERROR("Vertex data is empty.");
			return false;
		}

		m_Blob = vertexData;

		HRESULT hr = device->CreateVertexShader(m_Blob.data(), m_Blob.size(), NULL, m_Shader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create vertex shader from data.");
			return false;
		}

		ComPtr<ID3D11ShaderReflection> vertexShaderReflection;
		hr = D3DReflect(m_Blob.data(), m_Blob.size(), IID_ID3D11ShaderReflection, (void**)vertexShaderReflection.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get vertex shader reflection."));
			return false;
		}

		// Slots
		D3D11_SHADER_DESC vertexShaderDesc;
		vertexShaderReflection->GetDesc(&vertexShaderDesc);

		UINT resourceBindingCount = vertexShaderDesc.BoundResources;

		for (UINT i = 0; i < resourceBindingCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			vertexShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			uint8_t bindPoint = static_cast<uint8_t>(inputBindDesc.BindPoint);
			switch (inputBindDesc.Type)
			{
			case D3D_SIT_TEXTURE:
			{
				size_t textureHash = TO_HASH(String(inputBindDesc.Name));
				size_t texturePairIndex = UINT64_MAX;
				for (size_t j = 0; j < m_TextureSRVSamplerSlots.size(); ++j)
				{
					if (m_TextureSRVSamplerSlots[j].first == textureHash)
					{
						texturePairIndex = j;
						break;
					}
				}
				if (texturePairIndex == UINT64_MAX)
				{
					m_TextureSRVSamplerSlots.push_back(std::make_pair(textureHash, std::make_pair(bindPoint, UINT8_MAX)));
				}
				else
				{
					m_TextureSRVSamplerSlots[texturePairIndex].second.first = bindPoint;
				}
			}
			break;
			case D3D_SIT_CBUFFER:
				m_ConstantBufferSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), bindPoint));
				break;
			case D3D_SIT_STRUCTURED:
				m_BufferSRVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), bindPoint));
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
						samplerHash = TO_HASH(samplerName);
						size_t texturePairIndex = UINT64_MAX;
						for (size_t j = 0; j < m_TextureSRVSamplerSlots.size(); ++j)
						{
							if (m_TextureSRVSamplerSlots[j].first == samplerHash)
							{
								texturePairIndex = j;
								break;
							}
						}
						if (texturePairIndex == UINT64_MAX)
						{
							m_TextureSRVSamplerSlots.push_back(std::make_pair(samplerHash, std::make_pair(UINT8_MAX, bindPoint)));
						}
						else
						{
							m_TextureSRVSamplerSlots[texturePairIndex].second.second = bindPoint;
						}
					}
					else
					{
						BB_ERROR("Wrong sampler name.");
					}
				}
			}
			break;
			}
		}

		// Input layout
		UINT parameterCount = vertexShaderDesc.InputParameters;
		for (uint8_t i = 0; i < VERTEX_ATTRIBUTE_COUNT; ++i)
		{
			m_LayoutIndices[i] = UINT8_MAX;
		}

		m_Crc = 0;
		m_SemanticNames.resize(parameterCount);
		for (UINT i = 0; i < parameterCount; ++i)
		{
			D3D11_SIGNATURE_PARAMETER_DESC paramDesc;
			vertexShaderReflection->GetInputParameterDesc(i, &paramDesc);

			D3D11_INPUT_ELEMENT_DESC inputElementDesc = {};

			m_SemanticNames[i] = paramDesc.SemanticName;
			inputElementDesc.SemanticName = m_SemanticNames[i].c_str();
			inputElementDesc.SemanticIndex = paramDesc.SemanticIndex;
			
			uint32_t size;
			if (paramDesc.Mask == 1) 
			{
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32_FLOAT;
				size = 4;
			}
			else if (paramDesc.Mask <= 3) 
			{
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
				size = 8;
			}
			else if (paramDesc.Mask <= 7) 
			{
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
				size = 12;
			}
			else if (paramDesc.Mask <= 15) 
			{
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32A32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
				size = 16;
			}
			else
			{
				size = 0;
			}
			
			uint32_t nameInt = *reinterpret_cast<const uint32_t*>(paramDesc.SemanticName);
			m_Crc = CRCHelper::Calculate(nameInt, m_Crc);
			m_Crc = CRCHelper::Calculate(size, m_Crc);
			switch (nameInt)
			{
			case String4ToInt("POSI"): // POSITION
				m_LayoutIndices[0] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToInt("NORM"): // NORMAL
				m_LayoutIndices[1] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToInt("TANG"): // TANGENT
				m_LayoutIndices[2] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToInt("COLO"): // COLOR
				m_LayoutIndices[3] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToInt("TEXC"): // TEXCOORD
				m_LayoutIndices[4 + paramDesc.SemanticIndex] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToInt("BLEN"): // BLENDINDICES or BLENDWEIGHT
				nameInt = *reinterpret_cast<const uint32_t*>(paramDesc.SemanticName + 4);
				if (nameInt == String4ToInt("DWEI"))
				{
					m_LayoutIndices[8] = i;
				}
				else
				{
					m_LayoutIndices[9] = i;
				}
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToInt("REND"): // RENDER_INSTANCE
				inputElementDesc.InputSlot = 1;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_INSTANCE_DATA;
				inputElementDesc.InstanceDataStepRate = 1;
				break;
			case String4ToInt("SV_I"): // SV_InstanceID
				continue;
			}
			m_InputElementDescs.push_back(inputElementDesc);
		}
		m_Device = device;

		return true;
	}

	ID3D11InputLayout* GfxVertexShaderDX11::CreateLayout()
	{
		ID3D11InputLayout* layout;
		HRESULT hr = m_Device->CreateInputLayout(m_InputElementDescs.data(), static_cast<UINT>(m_InputElementDescs.size()), m_Blob.data(), m_Blob.size(), &layout);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating input layout."));
			return NULL;
		}
		return layout;
	}

	bool GfxGeometryShaderDX11::Initialize(ID3D11Device* device, const ByteData& geometryData)
	{
		if (geometryData.size() == 0)
		{
			BB_ERROR("Geometry data is empty.");
			return false;
		}

		HRESULT hr = device->CreateGeometryShader(geometryData.data(), geometryData.size(), NULL, m_Shader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create geometry shader from data.");
			return false;
		}

		ComPtr<ID3D11ShaderReflection> geometryShaderReflection;
		hr = D3DReflect(geometryData.data(), geometryData.size(), IID_ID3D11ShaderReflection, (void**)geometryShaderReflection.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get geometry shader reflection."));
			return false;
		}

		// Slots
		D3D11_SHADER_DESC geometryShaderDesc;
		geometryShaderReflection->GetDesc(&geometryShaderDesc);

		UINT constantBufferCount = geometryShaderDesc.ConstantBuffers;

		for (UINT i = 0; i < constantBufferCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			geometryShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			uint8_t bindPoint = static_cast<uint8_t>(inputBindDesc.BindPoint);
			if (inputBindDesc.Type == D3D_SIT_CBUFFER)
			{
				m_ConstantBufferSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), bindPoint));
			}
		}

		return true;
	}

	bool GfxFragmentShaderDX11::Initialize(ID3D11Device* device, const ByteData& fragmentData)
	{
		if (fragmentData.size() == 0)
		{
			BB_ERROR("Fragment data is empty.");
			return false;
		}

		HRESULT hr = device->CreatePixelShader(fragmentData.data(), fragmentData.size(), NULL, m_Shader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create fragment shader from data.");
			return false;
		}

		ComPtr<ID3D11ShaderReflection> pixelShaderReflection;
		hr = D3DReflect(fragmentData.data(), fragmentData.size(), IID_ID3D11ShaderReflection, (void**)pixelShaderReflection.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get pixel shader reflection."));
			return false;
		}

		// Slots
		D3D11_SHADER_DESC pixelShaderDesc;
		pixelShaderReflection->GetDesc(&pixelShaderDesc);

		UINT resourceBindingCount = pixelShaderDesc.BoundResources;

		for (UINT i = 0; i < resourceBindingCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			pixelShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			uint8_t bindPoint = static_cast<uint8_t>(inputBindDesc.BindPoint);
			switch (inputBindDesc.Type)
			{
			case D3D_SIT_TEXTURE:
			{
				size_t textureHash = TO_HASH(String(inputBindDesc.Name));
				size_t texturePairIndex = UINT64_MAX;
				for (size_t j = 0; j < m_TextureSRVSamplerSlots.size(); ++j)
				{
					if (m_TextureSRVSamplerSlots[j].first == textureHash)
					{
						texturePairIndex = j;
						break;
					}
				}
				if (texturePairIndex == UINT64_MAX)
				{
					m_TextureSRVSamplerSlots.push_back(std::make_pair(textureHash, std::make_pair(bindPoint, UINT8_MAX)));
				}
				else
				{
					m_TextureSRVSamplerSlots[texturePairIndex].second.first = bindPoint;
				}
			}
			break;
			case D3D_SIT_CBUFFER:
				m_ConstantBufferSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), bindPoint));
				break;
			case D3D_SIT_STRUCTURED:
				m_BufferSRVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), bindPoint));
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
						samplerHash = TO_HASH(samplerName);
						size_t texturePairIndex = UINT64_MAX;
						for (size_t j = 0; j < m_TextureSRVSamplerSlots.size(); ++j)
						{
							if (m_TextureSRVSamplerSlots[j].first == samplerHash)
							{
								texturePairIndex = j;
								break;
							}
						}
						if (texturePairIndex == UINT64_MAX)
						{
							m_TextureSRVSamplerSlots.push_back(std::make_pair(samplerHash, std::make_pair(UINT8_MAX, bindPoint)));
						}
						else
						{
							m_TextureSRVSamplerSlots[texturePairIndex].second.second = bindPoint;
						}
					}
					else
					{
						BB_ERROR("Wrong sampler name.");
					}
				}
			}
			break;
			}
		}

		return true;
	}
}