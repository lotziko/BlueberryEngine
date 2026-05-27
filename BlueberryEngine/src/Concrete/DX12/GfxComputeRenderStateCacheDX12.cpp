#include "GfxComputeRenderStateCacheDX12.h"

#include "GfxDeviceDX12.h"
#include "GfxComputeShaderDX12.h"
#include "GfxTextureDX12.h"
#include "GfxBufferDX12.h"

namespace Blueberry
{
	GfxComputeRenderStateCacheDX12::GfxComputeRenderStateCacheDX12(GfxDeviceDX12* device) : m_Device(device)
	{
	}

	GfxComputeRenderStateDX12 GfxComputeRenderStateCacheDX12::GetRenderState(GfxComputeShader* shader)
	{
		size_t key = reinterpret_cast<size_t>(shader);

		GfxComputeRenderStateDX12 renderState = {};
		auto it = m_PipelineBindingStates.find(key);
		if (it != m_PipelineBindingStates.end())
		{
			FillRenderState(shader, renderState, it->second.first, it->second.second);
		}
		else
		{
			auto dxComputeShader = static_cast<GfxComputeShaderDX12*>(shader);

			D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineStateDesc = {};
			pipelineStateDesc.pRootSignature = m_Device->GetComputeRootSignature();
			pipelineStateDesc.CS.pShaderBytecode = dxComputeShader->m_Blob.data();
			pipelineStateDesc.CS.BytecodeLength = dxComputeShader->m_Blob.size();
			pipelineStateDesc.NodeMask = 1;
			pipelineStateDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

			GfxComputePipelineStateDX12 pipelineState = {};
			GfxComputeBindingStateDX12 bindingState = {};

			m_Device->m_Device->CreateComputePipelineState(&pipelineStateDesc, IID_PPV_ARGS(&pipelineState.pipelineState));

			// Constant buffers
			for (size_t i = 0; i < dxComputeShader->m_ConstantBufferSlots.size(); ++i)
			{
				auto& bufferSlot = dxComputeShader->m_ConstantBufferSlots[i];
				for (size_t j = 0; j < m_Device->m_BindedBuffers.size(); ++j)
				{
					if (m_Device->m_BindedBuffers[j].first == bufferSlot.first)
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
					if (m_Device->m_BindedBuffers[j].first == srvSlot.first)
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
					if (m_Device->m_BindedTextures[j].first == srvSlot.first)
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
					if (m_Device->m_BindedBuffers[j].first == uavSlot.first)
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
					if (m_Device->m_BindedTextures[j].first == uavSlot.first)
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
					if (m_Device->m_BindedTextures[j].first == samplerSlot.first)
					{
						bindingState.samplers.push_back({ static_cast<uint32_t>(j), static_cast<uint8_t>(samplerSlot.second) });
						break;
					}
				}
			}

			m_PipelineBindingStates.insert_or_assign(key, std::make_pair(pipelineState, bindingState));
			FillRenderState(shader, renderState, pipelineState, bindingState);
		}
		return renderState;
	}

	void GfxComputeRenderStateCacheDX12::FillRenderState(GfxComputeShader* shader, GfxComputeRenderStateDX12& renderState, const GfxComputePipelineStateDX12& pipelineState, const GfxComputeBindingStateDX12& bindingState)
	{
		renderState.pipelineState = pipelineState.pipelineState;

		for (auto& binding : bindingState.cbvs)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::s_PointerCache.Get(m_Device->m_BindedBuffers[binding.bindingIndex].second);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			renderState.constantBuffers[binding.slotIndex] = dxBuffer->m_ConstantBufferView.GetCPU();
			renderState.constantBuffersCount = std::max(renderState.constantBuffersCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.bufferSrvs)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::s_PointerCache.Get(m_Device->m_BindedBuffers[binding.bindingIndex].second);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			renderState.shaderResourceViews[binding.slotIndex] = dxBuffer->m_ShaderResourceView.GetCPU();
			renderState.shaderResourceViewsCount = std::max(renderState.shaderResourceViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.textureSrvs)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::s_PointerCache.Get(m_Device->m_BindedTextures[binding.bindingIndex].second);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			renderState.shaderResourceViews[binding.slotIndex] = dxTexture->m_ShaderResourceView.GetCPU();
			renderState.shaderResourceViewsCount = std::max(renderState.shaderResourceViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.bufferUavs)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::s_PointerCache.Get(m_Device->m_BindedBuffers[binding.bindingIndex].second);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			uint8_t slotIndex = binding.slotIndex;
			renderState.resources[slotIndex] = dxBuffer->m_Resource.Get();
			renderState.states[slotIndex] = dxBuffer->m_State;
			renderState.unorderedAccessViews[binding.slotIndex] = dxBuffer->m_UnorderedAccessView.GetCPU();
			renderState.unorderedAccessViewsCount = std::max(renderState.unorderedAccessViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.textureUavs)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::s_PointerCache.Get(m_Device->m_BindedTextures[binding.bindingIndex].second);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			uint8_t slotIndex = binding.slotIndex;
			renderState.resources[slotIndex] = dxTexture->m_Resource.Get();
			renderState.states[slotIndex] = dxTexture->m_State;
			renderState.unorderedAccessViews[slotIndex] = dxTexture->m_UnorderedAccessView.GetCPU();
			renderState.unorderedAccessViewsCount = std::max(renderState.unorderedAccessViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.samplers)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::s_PointerCache.Get(m_Device->m_BindedTextures[binding.bindingIndex].second);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			uint8_t sampler = dxTexture->m_Sampler;
			if (sampler == UINT8_MAX)
			{
				sampler = m_Device->GetSampler(dxTexture->m_WrapMode, dxTexture->m_FilterMode);
				dxTexture->m_Sampler = sampler;
			}
			renderState.samplers[binding.slotIndex] = sampler;
			renderState.samplersCount = std::max(renderState.samplersCount, binding.slotIndex + 1u);
		}

		renderState.isValid = true;
	}
}