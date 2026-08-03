#include "GfxRayTracingShaderDX12.h"

#include "Concrete\Windows\DxcHelper.h"
#include "..\Windows\WindowsHelper.h"

#include <d3d12shader.h>

namespace Blueberry
{
	bool GfxRayTracingShaderDX12::Initialize(ID3D12Device* device, const ByteData& rayTracingData)
	{
		m_Blob = rayTracingData;

		ComPtr<IDxcBlobEncoding> blob;
		HRESULT hr = DxcHelper::GetLibrary()->CreateBlobWithEncodingFromPinned(rayTracingData.data(), static_cast<UINT32>(rayTracingData.size()), CP_UTF8, blob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create ray tracing shader blob."));
			return false;
		}

		ComPtr<IDxcContainerReflection> containerReflection;
		hr = DxcCreateInstance(CLSID_DxcContainerReflection, IID_PPV_ARGS(&containerReflection));

		if (FAILED(hr))
		{
			return false;
		}

		hr = containerReflection->Load(blob.Get());

		if (FAILED(hr))
		{
			return false;
		}

		UINT32 partIndex;
		hr = containerReflection->FindFirstPartKind(DXC_PART_DXIL, &partIndex);

		if (FAILED(hr))
		{
			return false;
		}

		ComPtr<ID3D12LibraryReflection> libraryReflection;
		hr = containerReflection->GetPartReflection(partIndex, IID_PPV_ARGS(&libraryReflection));
		
		D3D12_LIBRARY_DESC libraryDesc;
		libraryReflection->GetDesc(&libraryDesc);

		for (UINT i = 0; i < libraryDesc.FunctionCount; i++)
		{
			ID3D12FunctionReflection* function = libraryReflection->GetFunctionByIndex(i);

			D3D12_FUNCTION_DESC functionDesc;
			function->GetDesc(&functionDesc);

			String functionName = String(functionDesc.Name);
			
			auto raygenPos = functionName.find("RayGeneration");
			auto missPos = functionName.find("Miss");
			if (raygenPos != std::string::npos || missPos != std::string::npos)
			{
				UINT resourceBindingCount = functionDesc.BoundResources;

				for (UINT i = 0; i < resourceBindingCount; i++)
				{
					D3D12_SHADER_INPUT_BIND_DESC inputBindDesc;
					function->GetResourceBindingDesc(i, &inputBindDesc);
					switch (inputBindDesc.Type)
					{
					case D3D_SIT_TEXTURE:
						m_TextureSRVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
						break;
					case D3D_SIT_RTACCELERATIONSTRUCTURE:
						m_AccelerationStructureSlot = inputBindDesc.BindPoint;
						break;
					case D3D_SIT_CBUFFER:
						m_ConstantBufferSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
						break;
					case D3D11_SIT_UAV_RWTYPED:
						m_TextureUAVSlots.push_back(std::make_pair(TO_HASH(String(inputBindDesc.Name)), inputBindDesc.BindPoint));
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
						}
						m_SamplerSlots.push_back(std::make_pair(TO_HASH(samplerName), inputBindDesc.BindPoint));
					}
					break;
					}
				}
			}
			else
			{
				auto pos = functionName.find("ClosestHit");
				if (pos != std::string::npos)
				{
					UINT resourceBindingCount = functionDesc.BoundResources;

					for (UINT j = 0; j < resourceBindingCount; j++)
					{
						D3D12_SHADER_INPUT_BIND_DESC inputBindDesc;
						function->GetResourceBindingDesc(j, &inputBindDesc);
						if (inputBindDesc.Type == D3D_SIT_CBUFFER)
						{
							if (strcmp(inputBindDesc.Name, "RTPerMaterialData") == 0)
							{
								ID3D12ShaderReflectionConstantBuffer* materialBuffer = function->GetConstantBufferByIndex(j);
								D3D12_SHADER_BUFFER_DESC bufferDesc = {};
								materialBuffer->GetDesc(&bufferDesc);
								for (UINT k = 0; k < bufferDesc.Variables; ++k)
								{
									ID3D12ShaderReflectionVariable* variable = materialBuffer->GetVariableByIndex(k);
									D3D12_SHADER_VARIABLE_DESC variableDesc = {};
									variable->GetDesc(&variableDesc);
									String samplerName = String(variableDesc.Name);
									auto pos = samplerName.find("_Sampler");
									if (pos != std::string::npos)
									{
										samplerName.replace(pos, samplerName.length() - pos, "");
										size_t samplerHash = TO_HASH(samplerName);
										size_t texturePairIndex = UINT64_MAX;
										for (size_t l = 0; l < m_BindlessTextureSRVSamplerSlots.size(); ++l)
										{
											if (m_BindlessTextureSRVSamplerSlots[l].first == samplerHash)
											{
												texturePairIndex = l;
												break;
											}
										}
										if (texturePairIndex == UINT64_MAX)
										{
											m_BindlessTextureSRVSamplerSlots.push_back(std::make_pair(samplerHash, std::make_pair(UINT8_MAX, static_cast<uint8_t>(k))));
										}
										else
										{
											m_BindlessTextureSRVSamplerSlots[texturePairIndex].second.second = static_cast<uint8_t>(k);
										}
									}
									else
									{
										size_t textureHash = TO_HASH(String(variableDesc.Name));
										size_t texturePairIndex = UINT64_MAX;
										for (size_t l = 0; l < m_BindlessTextureSRVSamplerSlots.size(); ++l)
										{
											if (m_BindlessTextureSRVSamplerSlots[l].first == textureHash)
											{
												texturePairIndex = l;
												break;
											}
										}
										if (texturePairIndex == UINT64_MAX)
										{
											m_BindlessTextureSRVSamplerSlots.push_back(std::make_pair(textureHash, std::make_pair(static_cast<uint8_t>(k), UINT8_MAX)));
										}
										else
										{
											m_BindlessTextureSRVSamplerSlots[texturePairIndex].second.first = static_cast<uint8_t>(k);
										}
									}
								}
							}
						}
					}
				}
			}
		}

		return true;
	}
}