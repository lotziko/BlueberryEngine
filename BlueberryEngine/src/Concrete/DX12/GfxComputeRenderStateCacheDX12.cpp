#include "GfxComputeRenderStateCacheDX12.h"

#include "Blueberry\Graphics\ComputeShader.h"

#include "GfxDeviceDX12.h"
#include "GfxComputeShaderDX12.h"
#include "GfxTextureDX12.h"
#include "GfxBufferDX12.h"

namespace Blueberry
{
	GfxComputeRenderStateCacheDX12::GfxComputeRenderStateCacheDX12(GfxDeviceDX12* device) : m_Device(device)
	{
	}

	GfxComputeRenderStateDX12 GfxComputeRenderStateCacheDX12::GetRenderState(ComputeShader* shader, uint32_t kernelIndex)
	{
		GfxComputeShader* computeShader = shader->GetKernel(kernelIndex);
		size_t key = reinterpret_cast<size_t>(computeShader);

		GfxComputeRenderStateDX12 renderState = {};
		auto it = m_PipelineBindingStates.find(key);
		if (it != m_PipelineBindingStates.end())
		{
			FillRenderState(renderState, it->second.first, it->second.second);
		}
		else
		{
			auto dxComputeShader = static_cast<GfxComputeShaderDX12*>(computeShader);

			GfxComputePipelineStateDX12 pipelineState = {};
			GfxComputeBindingStateDX12 bindingState = {};

			D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineStateDesc = {};
			pipelineStateDesc.pRootSignature = m_Device->GetComputeRootSignature();
			pipelineStateDesc.CS.pShaderBytecode = dxComputeShader->m_Blob.data();
			pipelineStateDesc.CS.BytecodeLength = dxComputeShader->m_Blob.size();
			pipelineStateDesc.NodeMask = 1;
			pipelineStateDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

			m_Device->m_Device->CreateComputePipelineState(&pipelineStateDesc, IID_PPV_ARGS(&pipelineState.pipelineState));

			// Constant buffers
			for (size_t i = 0; i < dxComputeShader->m_ConstantBufferSlots.size(); ++i)
			{
				auto& bufferSlot = dxComputeShader->m_ConstantBufferSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedBuffers.size(); ++j)
				{
					if (m_Device->m_BindedBuffers[j].id == bufferSlot.first)
					{
						bindingState.cbvs.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(bufferSlot.second) });
						break;
					}
				}
			}

			// Structured buffer SRVs
			for (size_t i = 0; i < dxComputeShader->m_BufferSRVSlots.size(); ++i)
			{
				auto& srvSlot = dxComputeShader->m_BufferSRVSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedBuffers.size(); ++j)
				{
					if (m_Device->m_BindedBuffers[j].id == srvSlot.first)
					{
						bindingState.bufferSrvs.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(srvSlot.second) });
						break;
					}
				}
			}

			// Texture SRVs
			for (size_t i = 0; i < dxComputeShader->m_TextureSRVSlots.size(); ++i)
			{
				auto& srvSlot = dxComputeShader->m_TextureSRVSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedTextures.size(); ++j)
				{
					if (m_Device->m_BindedTextures[j].id == srvSlot.first)
					{
						bindingState.textureSrvs.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(srvSlot.second) });
						break;
					}
				}
			}

			// Structured buffers UAVs
			for (size_t i = 0; i < dxComputeShader->m_BufferUAVSlots.size(); ++i)
			{
				auto& uavSlot = dxComputeShader->m_BufferUAVSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedBuffers.size(); ++j)
				{
					if (m_Device->m_BindedBuffers[j].id == uavSlot.first)
					{
						bindingState.bufferUavs.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(uavSlot.second) });
						break;
					}
				}
			}

			// Texture UAVs
			for (size_t i = 0; i < dxComputeShader->m_TextureUAVSlots.size(); ++i)
			{
				auto& uavSlot = dxComputeShader->m_TextureUAVSlots[i];
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
			for (size_t i = 0; i < dxComputeShader->m_SamplerSlots.size(); ++i)
			{
				auto& samplerSlot = dxComputeShader->m_SamplerSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedTextures.size(); ++j)
				{
					if (m_Device->m_BindedTextures[j].id == samplerSlot.first)
					{
						bindingState.samplers.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(samplerSlot.second) });
						break;
					}
				}
			}

			// Static samplers
			for (size_t i = 0; i < dxComputeShader->m_StaticSamplerSlots.size(); ++i)
			{
				auto& samplerSlot = dxComputeShader->m_StaticSamplerSlots[i];
				bindingState.staticSamplers.push_back({ static_cast<uint8_t>(m_Device->GetSampler(std::get<1>(samplerSlot), std::get<0>(samplerSlot))), static_cast<uint8_t>(std::get<2>(samplerSlot)) });
			}

			m_PipelineBindingStates.insert_or_assign(key, std::make_pair(pipelineState, bindingState));
			FillRenderState(renderState, pipelineState, bindingState);
		}
		return renderState;
	}

	void GfxComputeRenderStateCacheDX12::FillRenderState(GfxComputeRenderStateDX12& renderState, const GfxComputePipelineStateDX12& pipelineState, const GfxComputeBindingStateDX12& bindingState)
	{
		renderState.pipelineState = pipelineState.pipelineState.Get();

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

		for (auto& binding : bindingState.bufferSrvs)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::Get(m_Device->m_BindedBuffers[binding.bindingIndex].index);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			dxBuffer->SetState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			renderState.shaderResourceViews[binding.slotIndex] = dxBuffer->GetShaderResourceView().GetCPU();
			renderState.shaderResourceViewsCount = std::max(renderState.shaderResourceViewsCount, binding.slotIndex + 1u);
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

		for (auto& binding : bindingState.bufferUavs)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::Get(m_Device->m_BindedBuffers[binding.bindingIndex].index);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			dxBuffer->SetUAVState();
			renderState.unorderedAccessViews[binding.slotIndex] = dxBuffer->GetUnorderedAccessView().GetCPU();
			renderState.unorderedAccessViewsCount = std::max(renderState.unorderedAccessViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.textureUavs)
		{
			auto& bindedTexture = m_Device->m_BindedTextures[binding.bindingIndex];
			GfxTextureDX12* dxTexture = GfxTextureDX12::Get(bindedTexture.index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			dxTexture->SetUAVState();
			renderState.unorderedAccessViews[binding.slotIndex] = bindedTexture.mip > 0 ? dxTexture->GetUnorderedAccessView(0, bindedTexture.mip).GetCPU() : dxTexture->GetUnorderedAccessView().GetCPU();
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

		for (auto& binding : bindingState.staticSamplers)
		{
			renderState.samplers[binding.slotIndex] = binding.sampler;
			renderState.samplersCount = std::max(renderState.samplersCount, binding.slotIndex + 1u);
		}

		renderState.isValid = true;
	}
}