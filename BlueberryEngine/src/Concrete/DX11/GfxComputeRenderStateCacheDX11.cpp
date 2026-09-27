#include "GfxComputeRenderStateCacheDX11.h"

#include "Blueberry\Graphics\ComputeShader.h"

#include "GfxDeviceDX11.h"
#include "GfxComputeShaderDX11.h"
#include "GfxTextureDX11.h"
#include "GfxBufferDX11.h"

namespace Blueberry
{
	GfxComputeRenderStateCacheDX11::GfxComputeRenderStateCacheDX11(GfxDeviceDX11* device) : m_Device(device)
	{
	}

	GfxComputeRenderStateDX11 GfxComputeRenderStateCacheDX11::GetRenderState(ComputeShader* shader, uint32_t kernelIndex)
	{
		GfxComputeShader* computeShader = shader->GetKernel(kernelIndex);
		size_t key = reinterpret_cast<size_t>(computeShader);

		GfxComputeRenderStateDX11 renderState = {};
		auto it = m_PipelineBindingStates.find(key);
		if (it != m_PipelineBindingStates.end())
		{
			FillRenderState(computeShader, renderState, it->second.first, it->second.second);
		}
		else
		{
			GfxComputeShaderDX11* dxComputeShader = static_cast<GfxComputeShaderDX11*>(computeShader);

			GfxComputePipelineStateDX11 pipelineState = {};
			GfxComputeBindingStateDX11 bindingState = {};

			pipelineState.computeShader = dxComputeShader->m_ComputeShader.Get();

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
				bindingState.staticSamplers.push_back({ m_Device->GetSamplerState(std::get<1>(samplerSlot), std::get<0>(samplerSlot)), static_cast<uint8_t>(std::get<2>(samplerSlot)) });
			}

			m_PipelineBindingStates.insert_or_assign(key, std::make_pair(pipelineState, bindingState));
			FillRenderState(computeShader, renderState, pipelineState, bindingState);
		}
		return renderState;
	}

	void GfxComputeRenderStateCacheDX11::FillRenderState(GfxComputeShader* shader, GfxComputeRenderStateDX11& renderState, const GfxComputePipelineStateDX11& pipelineState, const GfxComputeBindingStateDX11& bindingState)
	{
		renderState.computeShader = pipelineState.computeShader;

		for (auto& binding : bindingState.cbvs)
		{
			GfxBufferDX11* dxBuffer = GfxBufferDX11::Get(m_Device->m_BindedBuffers[binding.bindingIndex].index);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			renderState.constantBuffers[binding.slotIndex] = dxBuffer->GetBuffer();
			renderState.constantBuffersCount = std::max(renderState.constantBuffersCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.bufferSrvs)
		{
			GfxBufferDX11* dxBuffer = GfxBufferDX11::Get(m_Device->m_BindedBuffers[binding.bindingIndex].index);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			renderState.shaderResourceViews[binding.slotIndex] = dxBuffer->GetShaderResourceView();
			renderState.shaderResourceViewsCount = std::max(renderState.shaderResourceViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.textureSrvs)
		{
			auto& bindedTexture = m_Device->m_BindedTextures[binding.bindingIndex];
			GfxTextureDX11* dxTexture = GfxTextureDX11::Get(bindedTexture.index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			renderState.shaderResourceViews[binding.slotIndex] = bindedTexture.mip == UINT32_MAX ? dxTexture->GetShaderResourceView() : dxTexture->GetShaderResourceView(0, bindedTexture.mip);
			renderState.shaderResourceViewsCount = std::max(renderState.shaderResourceViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.bufferUavs)
		{
			GfxBufferDX11* dxBuffer = GfxBufferDX11::Get(m_Device->m_BindedBuffers[binding.bindingIndex].index);
			if (dxBuffer == nullptr)
			{
				BB_ERROR("Buffer is missing.");
				continue;
			}
			uint8_t slotIndex = binding.slotIndex;
			renderState.unorderedAccessViews[binding.slotIndex] = dxBuffer->GetUnorderedAccessView();
			renderState.unorderedAccessViewsCount = std::max(renderState.unorderedAccessViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.textureUavs)
		{
			auto& bindedTexture = m_Device->m_BindedTextures[binding.bindingIndex];
			GfxTextureDX11* dxTexture = GfxTextureDX11::Get(bindedTexture.index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			uint8_t slotIndex = binding.slotIndex;
			renderState.unorderedAccessViews[slotIndex] = bindedTexture.mip == UINT32_MAX ? dxTexture->GetUnorderedAccessView() : dxTexture->GetUnorderedAccessView(0, bindedTexture.mip);
			renderState.unorderedAccessViewsCount = std::max(renderState.unorderedAccessViewsCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.samplers)
		{
			GfxTextureDX11* dxTexture = GfxTextureDX11::Get(m_Device->m_BindedTextures[binding.bindingIndex].index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			ID3D11SamplerState* samplerState = dxTexture->GetSamplerState();
			if (samplerState == nullptr)
			{
				samplerState = m_Device->GetSamplerState(dxTexture->GetWrapMode(), dxTexture->GetFilterMode());
				dxTexture->SetSamplerState(samplerState);
			}
			renderState.samplerStates[binding.slotIndex] = samplerState;
			renderState.samplerStatesCount = std::max(renderState.samplerStatesCount, binding.slotIndex + 1u);
		}

		for (auto& binding : bindingState.staticSamplers)
		{
			renderState.samplerStates[binding.slotIndex] = binding.samplerState;
			renderState.samplerStatesCount = std::max(renderState.samplerStatesCount, binding.slotIndex + 1u);
		}

		renderState.isValid = true;
	}
}