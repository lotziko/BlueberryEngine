#include "GfxShaderDX12.h"

#include "Blueberry\Tools\CRCHelper.h"
#include "Concrete\Windows\DxcHelper.h"
#include "..\Windows\WindowsHelper.h"

#include <d3d12shader.h>

namespace Blueberry
{
	constexpr uint32_t String4ToIntDX12(const char* s)
	{
		return (static_cast<uint32_t>(s[3]) << 24) | (static_cast<uint32_t>(s[2]) << 16) | (static_cast<uint32_t>(s[1]) << 8) | static_cast<uint32_t>(s[0]);
	}

	bool GfxVertexShaderDX12::Initialize(ID3D12Device* device, const ByteData& vertexData)
	{
		if (vertexData.size() == 0)
		{
			BB_ERROR("Vertex data is empty.");
			return false;
		}

		m_Blob = vertexData;

		ComPtr<IDxcBlobEncoding> blob;
		HRESULT hr = DxcHelper::GetLibrary()->CreateBlobWithEncodingFromPinned(vertexData.data(), static_cast<UINT32>(vertexData.size()), CP_UTF8, blob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create vertex shader blob."));
			return false;
		}

		ComPtr<ID3D12ShaderReflection> vertexShaderReflection;
		hr = DxcHelper::Reflect(blob.Get(), IID_PPV_ARGS(&vertexShaderReflection));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get vertex shader reflection."));
			return false;
		}

		// Slots
		D3D12_SHADER_DESC vertexShaderDesc;
		vertexShaderReflection->GetDesc(&vertexShaderDesc);

		unsigned int resourceBindingCount = vertexShaderDesc.BoundResources;

		// TODO global samplers with UINT8_MAX in first pair value
		for (uint32_t i = 0; i < resourceBindingCount; i++)
		{
			D3D12_SHADER_INPUT_BIND_DESC inputBindDesc;
			vertexShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			uint8_t bindPoint = static_cast<uint8_t>(inputBindDesc.BindPoint);
			switch (inputBindDesc.Type)
			{
			case D3D_SIT_TEXTURE:
			{
				size_t textureHash = TO_HASH(String(inputBindDesc.Name));
				size_t texturePairIndex = UINT64_MAX;
				for (size_t i = 0; i < m_TextureSRVSamplerSlots.size(); ++i)
				{
					if (m_TextureSRVSamplerSlots[i].first == textureHash)
					{
						texturePairIndex = i;
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
				auto pos = samplerName.find("_Sampler");
				if (pos != std::string::npos)
				{
					samplerName.replace(pos, samplerName.length() - pos, "");
				}
				else
				{
					BB_ERROR("Wrong sampler name.");
					continue;
				}
				size_t samplerHash = TO_HASH(samplerName);
				size_t texturePairIndex = UINT64_MAX;
				for (size_t i = 0; i < m_TextureSRVSamplerSlots.size(); ++i)
				{
					if (m_TextureSRVSamplerSlots[i].first == samplerHash)
					{
						texturePairIndex = i;
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
			break;
			}
		}

		// Input layout
		uint32_t parameterCount = vertexShaderDesc.InputParameters;
		for (uint8_t i = 0; i < VERTEX_ATTRIBUTE_COUNT; ++i)
		{
			m_LayoutIndices[i] = UINT8_MAX;
		}

		m_Crc = 0;
		m_SemanticNames.resize(parameterCount);
		for (unsigned int i = 0; i < parameterCount; ++i)
		{
			D3D12_SIGNATURE_PARAMETER_DESC paramDesc;
			vertexShaderReflection->GetInputParameterDesc(i, &paramDesc);

			D3D12_INPUT_ELEMENT_DESC inputElementDesc = {};

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
			case String4ToIntDX12("POSI"): // POSITION
				m_LayoutIndices[0] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToIntDX12("NORM"): // NORMAL
				m_LayoutIndices[1] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToIntDX12("TANG"): // TANGENT
				m_LayoutIndices[2] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToIntDX12("COLO"): // COLOR
				m_LayoutIndices[3] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToIntDX12("TEXC"): // TEXCOORD
				m_LayoutIndices[4 + paramDesc.SemanticIndex] = i;
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToIntDX12("BLEN"): // BLENDINDICES or BLENDWEIGHT
				nameInt = *reinterpret_cast<const uint32_t*>(paramDesc.SemanticName + 4);
				if (nameInt == String4ToIntDX12("DWEI"))
				{
					m_LayoutIndices[8] = i;
				}
				else
				{
					m_LayoutIndices[9] = i;
				}
				inputElementDesc.InputSlot = 0;
				inputElementDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
				break;
			case String4ToIntDX12("REND"): // RENDER_INSTANCE
				inputElementDesc.InputSlot = 1;
				inputElementDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
				inputElementDesc.InstanceDataStepRate = 1;
				break;
			case String4ToIntDX12("SV_I"): // SV_InstanceID
				continue;
			}
			m_InputElementDescs.push_back(inputElementDesc);
		}
		return true;
	}

	bool GfxGeometryShaderDX12::Initialize(ID3D12Device* device, const ByteData& geometryData)
	{
		if (geometryData.size() == 0)
		{
			BB_ERROR("Geometry data is empty.");
			return false;
		}

		m_Blob = geometryData;

		ComPtr<IDxcBlobEncoding> blob;
		HRESULT hr = DxcHelper::GetLibrary()->CreateBlobWithEncodingFromPinned(geometryData.data(), static_cast<UINT32>(geometryData.size()), CP_UTF8, blob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create geometry shader blob."));
			return false;
		}

		ComPtr<ID3D12ShaderReflection> geometryShaderReflection;
		hr = DxcHelper::Reflect(blob.Get(), IID_PPV_ARGS(&geometryShaderReflection));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get geometry shader reflection."));
			return false;
		}

		// Slots
		D3D12_SHADER_DESC geometryShaderDesc;
		geometryShaderReflection->GetDesc(&geometryShaderDesc);

		unsigned int resourceBindingCount = geometryShaderDesc.BoundResources;

		for (uint32_t i = 0; i < resourceBindingCount; i++)
		{
			D3D12_SHADER_INPUT_BIND_DESC inputBindDesc;
			geometryShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			uint8_t bindPoint = static_cast<uint8_t>(inputBindDesc.BindPoint);
			if (inputBindDesc.Type == D3D_SIT_CBUFFER)
			{
				m_ConstantBufferSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), bindPoint));
			}
		}
		return true;
	}

	bool GfxFragmentShaderDX12::Initialize(ID3D12Device* device, const ByteData& fragmentData)
	{
		if (fragmentData.size() == 0)
		{
			BB_ERROR("Fragment data is empty.");
			return false;
		}

		m_Blob = fragmentData;

		ComPtr<IDxcBlobEncoding> blob;
		HRESULT hr = DxcHelper::GetLibrary()->CreateBlobWithEncodingFromPinned(fragmentData.data(), static_cast<UINT32>(fragmentData.size()), CP_UTF8, blob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create pixel shader blob."));
			return false;
		}

		ComPtr<ID3D12ShaderReflection> pixelShaderReflection;
		hr = DxcHelper::Reflect(blob.Get(), IID_PPV_ARGS(&pixelShaderReflection));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get pixel shader reflection."));
			return false;
		}
		
		// Slots
		D3D12_SHADER_DESC pixelShaderDesc;
		pixelShaderReflection->GetDesc(&pixelShaderDesc);

		unsigned int resourceBindingCount = pixelShaderDesc.BoundResources;

		for (uint32_t i = 0; i < resourceBindingCount; i++)
		{
			D3D12_SHADER_INPUT_BIND_DESC inputBindDesc;
			pixelShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			uint8_t bindPoint = static_cast<uint8_t>(inputBindDesc.BindPoint);
			switch (inputBindDesc.Type)
			{
			case D3D_SIT_TEXTURE:
			{
				size_t textureHash = TO_HASH(String(inputBindDesc.Name));
				size_t texturePairIndex = UINT64_MAX;
				for (size_t i = 0; i < m_TextureSRVSamplerSlots.size(); ++i)
				{
					if (m_TextureSRVSamplerSlots[i].first == textureHash)
					{
						texturePairIndex = i;
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
				auto pos = samplerName.find("_Sampler");
				if (pos != std::string::npos)
				{
					samplerName.replace(pos, samplerName.length() - pos, "");
				}
				else
				{
					BB_ERROR("Wrong sampler name.");
					continue;
				}
				size_t samplerHash = TO_HASH(samplerName);
				size_t texturePairIndex = UINT64_MAX;
				for (size_t i = 0; i < m_TextureSRVSamplerSlots.size(); ++i)
				{
					if (m_TextureSRVSamplerSlots[i].first == samplerHash)
					{
						texturePairIndex = i;
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
			break;
			}
		}
		return true;
	}
}