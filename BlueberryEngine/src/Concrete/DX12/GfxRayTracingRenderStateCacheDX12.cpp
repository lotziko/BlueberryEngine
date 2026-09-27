#include "GfxRayTracingRenderStateCacheDX12.h"

#include "Blueberry\Graphics\RayTracingShader.h"
#include "Blueberry\Graphics\Material.h"

#include "GfxDeviceDX12.h"
#include "GfxRayTracingShaderDX12.h"
#include "GfxTopLevelAccelerationStructureDX12.h"
#include "GfxBufferDX12.h"
#include "GfxTextureDX12.h"
#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	static GfxRayTracingRenderStateDX12 s_DefaultRayTracingRenderState = {};

	GfxRayTracingRenderStateCacheDX12::GfxRayTracingRenderStateCacheDX12(GfxDeviceDX12* device) : m_Device(device)
	{
		std::fill_n(s_DefaultRayTracingRenderState.constantBuffers, _countof(s_DefaultRayTracingRenderState.constantBuffers), device->m_EmptyCbv.GetCPU());
		std::fill_n(s_DefaultRayTracingRenderState.shaderResourceViews, _countof(s_DefaultRayTracingRenderState.shaderResourceViews), device->m_EmptySrv.GetCPU());
		std::fill_n(s_DefaultRayTracingRenderState.unorderedAccessViews, _countof(s_DefaultRayTracingRenderState.unorderedAccessViews), device->m_EmptyUav.GetCPU());
	}

	GfxRayTracingRenderStateDX12 GfxRayTracingRenderStateCacheDX12::GetRenderState(RayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure)
	{
		GfxRayTracingShader* rayTracingShader = shader->Get();
		size_t key = reinterpret_cast<size_t>(rayTracingShader);

		GfxRayTracingRenderStateDX12 renderState = s_DefaultRayTracingRenderState;
		auto it = m_PipelineBindingStates.find(key);
		if (it != m_PipelineBindingStates.end())
		{
			FillShaderBindingTable(rayTracingShader, accelerationStructure, renderState, it->second.first, it->second.second);
			FillRenderState(rayTracingShader, accelerationStructure, renderState, it->second.first, it->second.second);
		}
		else
		{
			const RayTracingShaderData& shaderData = shader->GetData();
			GfxRayTracingShaderDX12* dxRayTracingShader = static_cast<GfxRayTracingShaderDX12*>(rayTracingShader);

			GfxRayTracingPipelineStateDX12 pipelineState = {};
			GfxRayTracingBindingStateDX12 bindingState = {};

			CD3DX12_STATE_OBJECT_DESC raytracingPipeline(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);
			auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
			D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE(dxRayTracingShader->m_Blob.data(), dxRayTracingShader->m_Blob.size());
			lib->SetDXILLibrary(&libdxil);
			lib->DefineExport(L"RayGeneration");
			lib->DefineExport(L"AnyHit0");
			lib->DefineExport(L"ClosestHit0");
			lib->DefineExport(L"Miss0");

			auto hitGroup0 = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
			hitGroup0->SetClosestHitShaderImport(L"ClosestHit0");
			hitGroup0->SetHitGroupExport(L"HitGroup0");
			hitGroup0->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

			auto hitGroup1 = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
			hitGroup1->SetClosestHitShaderImport(L"ClosestHit0");
			hitGroup1->SetAnyHitShaderImport(L"AnyHit0");
			hitGroup1->SetHitGroupExport(L"HitGroup1");
			hitGroup1->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

			auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
			shaderConfig->Config(shaderData.GetPayloadSize(), shaderData.GetAttributesSize());

			auto localRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
			localRootSignature->SetRootSignature(m_Device->m_DxrLocalRootSignature.Get());

			auto rootSignatureAssociation = raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
			rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
			rootSignatureAssociation->AddExport(L"HitGroup0");
			rootSignatureAssociation->AddExport(L"HitGroup1");

			auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
			globalRootSignature->SetRootSignature(m_Device->m_DxrGlobalRootSignature.Get());

			auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
			pipelineConfig->Config(shaderData.GetRayRecursionDepth());

			HRESULT hr = m_Device->m_DxrDevice->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&pipelineState.stateObject));

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating dxr state object."));
				return {};
			}

			hr = pipelineState.stateObject.As<ID3D12StateObjectProperties>(&pipelineState.stateObjectProperties);

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error getting dxr state object properties."));
				return {};
			}

			// Constant buffers
			for (size_t i = 0; i < dxRayTracingShader->m_ConstantBufferSlots.size(); ++i)
			{
				auto& bufferSlot = dxRayTracingShader->m_ConstantBufferSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedBuffers.size(); ++j)
				{
					if (m_Device->m_BindedBuffers[j].id == bufferSlot.first)
					{
						bindingState.cbvs.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(bufferSlot.second) });
						break;
					}
				}
			}

			// Texture SRVs
			for (size_t i = 0; i < dxRayTracingShader->m_TextureSRVSlots.size(); ++i)
			{
				auto& srvSlot = dxRayTracingShader->m_TextureSRVSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedTextures.size(); ++j)
				{
					if (m_Device->m_BindedTextures[j].id == srvSlot.first)
					{
						bindingState.textureSrvs.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(srvSlot.second) });
						break;
					}
				}
			}

			// Texture UAVs
			for (size_t i = 0; i < dxRayTracingShader->m_TextureUAVSlots.size(); ++i)
			{
				auto& uavSlot = dxRayTracingShader->m_TextureUAVSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedTextures.size(); ++j)
				{
					if (m_Device->m_BindedTextures[j].id == uavSlot.first)
					{
						bindingState.textureUavs.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(uavSlot.second) });
						break;
					}
				}
			}

			// Samplers
			for (size_t i = 0; i < dxRayTracingShader->m_SamplerSlots.size(); ++i)
			{
				auto& samplerSlot = dxRayTracingShader->m_SamplerSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedTextures.size(); ++j)
				{
					if (m_Device->m_BindedTextures[j].id == samplerSlot.first)
					{
						bindingState.samplers.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(samplerSlot.second) });
						break;
					}
				}
			}

			pipelineState.rayGenerationShaderTable = GfxRayTracingShaderTableDX12(m_Device);
			pipelineState.hitGroupShaderTable = GfxRayTracingShaderTableDX12(m_Device);
			pipelineState.missShaderTable = GfxRayTracingShaderTableDX12(m_Device);

			FillShaderBindingTable(rayTracingShader, accelerationStructure, renderState, pipelineState, bindingState);
			FillRenderState(rayTracingShader, accelerationStructure, renderState, pipelineState, bindingState);
			m_PipelineBindingStates.insert_or_assign(key, std::make_pair(std::move(pipelineState), std::move(bindingState)));
		}
		return renderState;
	}

	void GfxRayTracingRenderStateCacheDX12::FillShaderBindingTable(GfxRayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, const GfxRayTracingRenderStateDX12& renderState, GfxRayTracingPipelineStateDX12& pipelineState, GfxRayTracingBindingStateDX12& bindingState)
	{
		auto dxRayTracingShader = static_cast<GfxRayTracingShaderDX12*>(shader);
		auto dxAccelerationStructure = static_cast<GfxTopLevelAccelerationStructureDX12*>(accelerationStructure);

		pipelineState.rayGenerationShaderTable.Clear();
		pipelineState.rayGenerationShaderTable.AddRecord(pipelineState.stateObjectProperties->GetShaderIdentifier(L"RayGeneration"), D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, nullptr, 0);
		pipelineState.rayGenerationShaderTable.Build();

		struct GfxRayTracingRecordStateDX12
		{
			D3D12_GPU_VIRTUAL_ADDRESS vertexBufferAddress;
			D3D12_GPU_VIRTUAL_ADDRESS indexBufferAddress;
			UINT vertexStride;
			UINT normalOffset;
			UINT uv0Offset;
			UINT bindlessIndexes[16];
		};

		pipelineState.hitGroupShaderTable.Clear();
		void* hitGroup0Identifier = pipelineState.stateObjectProperties->GetShaderIdentifier(L"HitGroup0");
		void* hitGroup1Identifier = pipelineState.stateObjectProperties->GetShaderIdentifier(L"HitGroup1");
		for (size_t i = 0; i < dxAccelerationStructure->m_InstanceDescs.size(); ++i)
		{
			// TODO ray types for instanceDesc.InstanceContributionToHitGroupIndex
			D3D12_RAYTRACING_INSTANCE_DESC& instanceDesc = dxAccelerationStructure->m_InstanceDescs[i];
			for (UINT j = 0; j < instanceDesc.InstanceID; ++j)
			{
				GfxRayTracingInstanceDataDX12& instanceData = dxAccelerationStructure->m_InstanceDatas[instanceDesc.InstanceContributionToHitGroupIndex + j];
				auto it = bindingState.materialBindings.find(instanceData.material->GetObjectId());
				if (it != bindingState.materialBindings.end())
				{
					GfxRayTracingRecordStateDX12 recordState = {};
					recordState.vertexBufferAddress = instanceData.geometryData.vertexBufferAddress;
					recordState.indexBufferAddress = instanceData.geometryData.indexBufferAddress;
					recordState.vertexStride = instanceData.geometryData.vertexStride;
					recordState.normalOffset = instanceData.geometryData.normalOffset;
					recordState.uv0Offset = instanceData.geometryData.uv0Offset;

					GfxRayTracingMaterialBindingDX12& materialBinding = it->second;
					for (auto& texture : materialBinding.bindlessTextures)
					{
						GfxTextureDX12* dxTexture = GfxTextureDX12::Get(GetTextureIndex(instanceData.material, texture.bindingIndex));
						if (dxTexture == nullptr)
						{
							BB_ERROR("Texture is missing.");
							continue;
						}
						dxTexture->SetState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
						recordState.bindlessIndexes[texture.srvSlot] = dxTexture->GetRingShaderResourceView().GetIndex();
						if (texture.samplerSlot != UINT8_MAX)
						{
							uint8_t sampler = dxTexture->GetSampler();
							if (sampler == UINT8_MAX)
							{
								sampler = m_Device->GetSampler(dxTexture->GetWrapMode(), dxTexture->GetFilterMode());
								dxTexture->SetSampler(sampler);
							}
							recordState.bindlessIndexes[texture.samplerSlot] = sampler;
						}
					}
					pipelineState.hitGroupShaderTable.AddRecord(instanceData.material->IsOpaque() ? hitGroup0Identifier : hitGroup1Identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, &recordState, sizeof(GfxRayTracingRecordStateDX12));
				}
				else
				{
					GfxRayTracingMaterialBindingDX12 materialBinding = {};

					// Bindless material textures
					for (auto it = dxRayTracingShader->m_BindlessTextureSRVSamplerSlots.begin(); it != dxRayTracingShader->m_BindlessTextureSRVSamplerSlots.end(); it++)
					{
						uint32_t offset = GetTextureSlot(instanceData.material, it->first);
						if (offset != UINT32_MAX)
						{
							materialBinding.bindlessTextures.push_back({ offset, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
						}
					}

					GfxRayTracingRecordStateDX12 recordState = {};
					recordState.vertexBufferAddress = instanceData.geometryData.vertexBufferAddress;
					recordState.indexBufferAddress = instanceData.geometryData.indexBufferAddress;
					recordState.vertexStride = instanceData.geometryData.vertexStride;
					recordState.normalOffset = instanceData.geometryData.normalOffset;
					recordState.uv0Offset = instanceData.geometryData.uv0Offset;

					for (auto& texture : materialBinding.bindlessTextures)
					{
						GfxTextureDX12* dxTexture = GfxTextureDX12::Get(GetTextureIndex(instanceData.material, texture.bindingIndex));
						if (dxTexture == nullptr)
						{
							BB_ERROR("Texture is missing.");
							continue;
						}
						dxTexture->SetState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
						recordState.bindlessIndexes[texture.srvSlot] = dxTexture->GetRingShaderResourceView().GetIndex();
						if (texture.samplerSlot != UINT8_MAX)
						{
							uint8_t sampler = dxTexture->GetSampler();
							if (sampler == UINT8_MAX)
							{
								sampler = m_Device->GetSampler(dxTexture->GetWrapMode(), dxTexture->GetFilterMode());
								dxTexture->SetSampler(sampler);
							}
							recordState.bindlessIndexes[texture.samplerSlot] = sampler;
						}
					}
					pipelineState.hitGroupShaderTable.AddRecord(instanceData.material->IsOpaque() ? hitGroup0Identifier : hitGroup1Identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, &recordState, sizeof(GfxRayTracingRecordStateDX12));
					bindingState.materialBindings.insert_or_assign(instanceData.material->GetObjectId(), std::move(materialBinding));
				}
			}
		}
		pipelineState.hitGroupShaderTable.Build();

		pipelineState.missShaderTable.Clear();
		pipelineState.missShaderTable.AddRecord(pipelineState.stateObjectProperties->GetShaderIdentifier(L"Miss0"), D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, nullptr, 0);
		pipelineState.missShaderTable.Build();

		dxAccelerationStructure->Build();
	}

	void GfxRayTracingRenderStateCacheDX12::FillRenderState(GfxRayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, GfxRayTracingRenderStateDX12& renderState, const GfxRayTracingPipelineStateDX12& pipelineState, const GfxRayTracingBindingStateDX12& bindingState)
	{
		renderState.stateObject = pipelineState.stateObject.Get();
		renderState.stateObjectProperties = pipelineState.stateObjectProperties.Get();

		auto dxRayTracingShader = static_cast<GfxRayTracingShaderDX12*>(shader);
		auto dxAccelerationStructure = static_cast<GfxTopLevelAccelerationStructureDX12*>(accelerationStructure);

		renderState.shaderResourceViews[dxRayTracingShader->m_AccelerationStructureSlot] = dxAccelerationStructure->GetShaderResourceView().GetCPU();
		renderState.shaderResourceViewsCount = std::max(renderState.shaderResourceViewsCount, dxRayTracingShader->m_AccelerationStructureSlot + 1u);

		for (auto& binding : bindingState.cbvs)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::Get(m_Device->m_BindedBuffers[binding.bindingIndex].index);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			dxBuffer->SetState(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
			renderState.constantBuffers[binding.slotIndex] = dxBuffer->GetConstantBufferView().GetCPU();
			renderState.constantBuffersCount = std::max(renderState.constantBuffersCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.textureSrvs)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::Get(m_Device->m_BindedTextures[binding.bindingIndex].index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			dxTexture->SetState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			renderState.shaderResourceViews[binding.slotIndex] = dxTexture->GetShaderResourceView().GetCPU();
			renderState.shaderResourceViewsCount = std::max(renderState.shaderResourceViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.textureUavs)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::Get(m_Device->m_BindedTextures[binding.bindingIndex].index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			dxTexture->SetUAVState();
			renderState.unorderedAccessViews[binding.slotIndex] = dxTexture->GetUnorderedAccessView().GetCPU();
			renderState.unorderedAccessViewsCount = std::max(renderState.unorderedAccessViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.samplers)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::Get(m_Device->m_BindedTextures[binding.bindingIndex].index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			uint8_t sampler = dxTexture->GetSampler();
			if (sampler == UINT8_MAX)
			{
				sampler = m_Device->GetSampler(dxTexture->GetWrapMode(), dxTexture->GetFilterMode());
				dxTexture->SetSampler(sampler);
			}
			renderState.samplers[binding.slotIndex] = sampler;
			renderState.samplersCount = std::max(renderState.samplersCount, binding.slotIndex + 1u);
		}

		renderState.rayGenerationShaderTableAddress = pipelineState.rayGenerationShaderTable.GetResource()->GetGPUVirtualAddress();
		renderState.hitGroupShaderTableAddress = pipelineState.hitGroupShaderTable.GetResource()->GetGPUVirtualAddress();
		renderState.missShaderTableAddress = pipelineState.missShaderTable.GetResource()->GetGPUVirtualAddress();

		renderState.rayGenerationShaderTableSize = pipelineState.rayGenerationShaderTable.GetSize();
		renderState.hitGroupShaderTableSize = pipelineState.hitGroupShaderTable.GetSize();
		renderState.hitGroupShaderTableStride = renderState.hitGroupShaderTableSize / pipelineState.hitGroupShaderTable.GetRecordCount();
		renderState.missShaderTableSize = pipelineState.missShaderTable.GetSize();
	}
}