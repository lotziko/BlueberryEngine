#include "GfxRenderStateCacheDX12.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\VertexLayout.h"

#include "GfxDeviceDX12.h"
#include "GfxShaderDX12.h"
#include "GfxTextureDX12.h"
#include "GfxBufferDX12.h"

namespace Blueberry
{
	#define SHADER_RESOURCE_STATE (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)

	bool GfxPipelineStateKeyDX12::operator==(const GfxPipelineStateKeyDX12& other) const
	{
		return memcmp(this, &other, sizeof(GfxPipelineStateKeyDX12)) == 0;
	}

	bool GfxPipelineStateKeyDX12::operator!=(const GfxPipelineStateKeyDX12& other) const
	{
		return memcmp(this, &other, sizeof(GfxPipelineStateKeyDX12)) != 0;
	}

	bool GfxRenderStateKeyDX12::operator==(const GfxRenderStateKeyDX12& other) const
	{
		return memcmp(this, &other, sizeof(GfxRenderStateKeyDX12)) == 0;
	}

	bool GfxRenderStateKeyDX12::operator!=(const GfxRenderStateKeyDX12& other) const
	{
		return memcmp(this, &other, sizeof(GfxRenderStateKeyDX12)) != 0;
	}

	GfxRenderStateCacheDX12::GfxRenderStateCacheDX12(GfxDeviceDX12* device) : m_Device(device)
	{
	}

	D3D12_BLEND GetBlend(BlendMode blend)
	{
		switch (blend)
		{
		case BlendMode::One: return D3D12_BLEND_ONE;
		case BlendMode::Zero: return D3D12_BLEND_ZERO;
		case BlendMode::SrcAlpha: return D3D12_BLEND_SRC_ALPHA;
		case BlendMode::OneMinusSrcAlpha: return D3D12_BLEND_INV_SRC_ALPHA;
		default: return D3D12_BLEND_ONE;
		}
	}

	D3D12_PRIMITIVE_TOPOLOGY_TYPE GetPrimitiveTopology(Topology topology)
	{
		switch (topology)
		{
		case Topology::Unknown: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
		case Topology::PointList: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
		case Topology::LineList: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
		case Topology::LineStrip: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
		case Topology::TriangleList: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		default: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
		}
	}

	GfxRenderStateDX12 GfxRenderStateCacheDX12::GetRenderState(Material* material, uint64_t passId, VertexLayout* meshLayout, GfxTargetInfoDX12& targetInfo, Topology topology, uint32_t depthBias, float slopeDepthBias, bool isCounterClockwise, bool isSolid)
	{
		uint64_t keywordMask = static_cast<uint64_t>(Shader::GetActiveKeywordsMask()) | (static_cast<uint64_t>(material->GetActiveKeywordsMask()) << 32);
		ObjectId shaderObjectId = material->GetShader()->GetObjectId();
		ObjectId materialObjectId = material->GetObjectId();
		uint32_t meshLayoutCrc = meshLayout->GetCrc();
		uint32_t materialCrc = material->GetCRC();
		
		GfxRenderStateDX12 renderState = {};
		GfxPassData passData = {};
		GfxPipelineStateKeyDX12 pipelineStateKey = { keywordMask, passId, shaderObjectId, meshLayoutCrc, targetInfo, static_cast<uint32_t>(topology), depthBias, slopeDepthBias, isCounterClockwise, isSolid };
		GfxRenderStateKeyDX12 bindingStateKey = { keywordMask, passId, materialObjectId };
		auto psIt = m_PipelineStates.find(pipelineStateKey);
		auto bsIt = m_BindingStates.find(bindingStateKey);
		bool hasPipelineState = psIt != m_PipelineStates.end() && materialCrc == psIt->second.crc;
		bool hasBindingState = bsIt != m_BindingStates.end() && materialCrc == bsIt->second.crc;

		if (!hasPipelineState || !hasBindingState)
		{
			passData = GetPassData(material, passId);
		}

		if (hasBindingState)
		{
			FillRenderState(material, renderState, bsIt->second);
		}
		else
		{
			renderState.isValid = passData.isValid;

			if (!passData.isValid)
			{
				return renderState;
			}

			auto dxVertexShader = static_cast<GfxVertexShaderDX12*>(passData.vertexShader);
			auto dxGeometryShader = static_cast<GfxGeometryShaderDX12*>(passData.geometryShader);
			auto dxFragmentShader = static_cast<GfxFragmentShaderDX12*>(passData.fragmentShader);

			GfxBindingStateDX12 bindingState = {};
			bindingState.crc = materialCrc;
			List<size_t> usedTextures = {};

			// Vertex global constant buffers
			for (auto it = dxVertexShader->m_ConstantBufferSlots.begin(); it != dxVertexShader->m_ConstantBufferSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
				{
					if (it1->first == it->first)
					{
						bindingState.vertexBuffers.push_back({ offset, true, it->second, UINT8_MAX });
						break;
					}
				}
			}

			// Vertex global structured buffers
			for (auto it = dxVertexShader->m_BufferSRVSlots.begin(); it != dxVertexShader->m_BufferSRVSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
				{
					if (it1->first == it->first)
					{
						bindingState.vertexBuffers.push_back({ offset, true, UINT8_MAX, it->second });
						break;
					}
				}
			}

			// Vertex material textures
			for (auto it = dxVertexShader->m_TextureSRVSamplerSlots.begin(); it != dxVertexShader->m_TextureSRVSamplerSlots.end(); it++)
			{
				uint32_t offset = GetTextureIndex(material, it->first);
				if (offset != UINT32_MAX)
				{
					usedTextures.push_back(it->first);
					bindingState.vertexTextures.push_back({ offset, false, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
				}
			}

			// Vertex global textures
			for (auto it = dxVertexShader->m_TextureSRVSamplerSlots.begin(); it != dxVertexShader->m_TextureSRVSamplerSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedTextures.begin(); it1 < m_Device->m_BindedTextures.end(); ++it1, ++offset)
				{
					if (it1->first == it->first)
					{
						if (std::find(usedTextures.begin(), usedTextures.end(), it1->first) == usedTextures.end())
						{
							bindingState.vertexTextures.push_back({ offset, true, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
						}
						break;
					}
				}
			}
			usedTextures.clear();

			if (dxGeometryShader != nullptr)
			{
				// Geometry global constant buffers
				for (auto it = dxGeometryShader->m_ConstantBufferSlots.begin(); it != dxGeometryShader->m_ConstantBufferSlots.end(); it++)
				{
					uint32_t offset = 0;
					for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
					{
						if (it1->first == it->first)
						{
							bindingState.geometryBuffers.push_back({ offset, true, it->second, UINT8_MAX });
							break;
						}
					}
				}
			}

			// Fragment global constant buffers
			for (auto it = dxFragmentShader->m_ConstantBufferSlots.begin(); it != dxFragmentShader->m_ConstantBufferSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
				{
					if (it1->first == it->first)
					{
						bindingState.pixelBuffers.push_back({ offset, true, it->second, UINT8_MAX });
						break;
					}
				}
			}

			// Fragment global structured buffers
			for (auto it = dxFragmentShader->m_BufferSRVSlots.begin(); it != dxFragmentShader->m_BufferSRVSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
				{
					if (it1->first == it->first)
					{
						bindingState.pixelBuffers.push_back({ offset, true, UINT8_MAX, it->second });
						break;
					}
				}
			}

			// Fragment material textures
			for (auto it = dxFragmentShader->m_TextureSRVSamplerSlots.begin(); it != dxFragmentShader->m_TextureSRVSamplerSlots.end(); it++)
			{
				uint32_t offset = GetTextureIndex(material, it->first);
				if (offset != UINT32_MAX)
				{
					usedTextures.push_back(it->first);
					bindingState.pixelTextures.push_back({ offset, false, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
				}
			}

			// Fragment global textures
			for (auto it = dxFragmentShader->m_TextureSRVSamplerSlots.begin(); it != dxFragmentShader->m_TextureSRVSamplerSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedTextures.begin(); it1 < m_Device->m_BindedTextures.end(); ++it1, ++offset)
				{
					if (it1->first == it->first)
					{
						if (std::find(usedTextures.begin(), usedTextures.end(), it1->first) == usedTextures.end())
						{
							bindingState.pixelTextures.push_back({ offset, true, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
						}
						break;
					}
				}
			}

			m_BindingStates.insert_or_assign(bindingStateKey, bindingState);
			FillRenderState(material, renderState, bindingState);
		}

		if (hasPipelineState)
		{
			renderState.pipelineState = psIt->second.pipelineState;
			renderState.isValid = true;
		}
		else
		{
			GfxPipelineStateDX12 pipelineState = {};
			pipelineState.crc = materialCrc;
			pipelineState.isValid = passData.isValid;

			if (!passData.isValid)
			{
				return renderState;
			}

			auto dxVertexShader = static_cast<GfxVertexShaderDX12*>(passData.vertexShader);
			auto dxGeometryShader = static_cast<GfxGeometryShaderDX12*>(passData.geometryShader);
			auto dxFragmentShader = static_cast<GfxFragmentShaderDX12*>(passData.fragmentShader);

			D3D12_GRAPHICS_PIPELINE_STATE_DESC pipelineStateDesc = {};
			pipelineStateDesc.pRootSignature = m_Device->GetGraphicsRootSignature();
			pipelineStateDesc.VS.pShaderBytecode = dxVertexShader->m_Blob.data();
			pipelineStateDesc.VS.BytecodeLength = dxVertexShader->m_Blob.size();
			pipelineStateDesc.PS.pShaderBytecode = dxFragmentShader->m_Blob.data();
			pipelineStateDesc.PS.BytecodeLength = dxFragmentShader->m_Blob.size();
			if (dxGeometryShader != nullptr)
			{
				pipelineStateDesc.GS.pShaderBytecode = dxGeometryShader->m_Blob.data();
				pipelineStateDesc.GS.BytecodeLength = dxGeometryShader->m_Blob.size();
			}
			pipelineStateDesc.SampleMask = UINT_MAX;
			pipelineStateDesc.PrimitiveTopologyType = GetPrimitiveTopology(topology);
			pipelineStateDesc.NumRenderTargets = targetInfo.renderTargetFormat == DXGI_FORMAT_UNKNOWN ? 0 : 1;
			pipelineStateDesc.RTVFormats[0] = targetInfo.renderTargetFormat;
			pipelineStateDesc.DSVFormat = targetInfo.depthStencilFormat;
			pipelineStateDesc.SampleDesc.Count = targetInfo.sampleCount;
			pipelineStateDesc.SampleDesc.Quality = targetInfo.sampleQuality;
			pipelineStateDesc.NodeMask = 1;
			pipelineStateDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

			for (uint32_t i = 0; i < RENDERABLE_VERTEX_ATTRIBUTE_COUNT; ++i)
			{
				uint32_t offset = meshLayout->GetOffset(i);
				uint8_t index = dxVertexShader->m_LayoutIndices[i];
				if (index != UINT8_MAX)
				{
					dxVertexShader->m_InputElementDescs[index].AlignedByteOffset = offset;
				}
			}

			D3D12_INPUT_LAYOUT_DESC& inputLayoutDesc = pipelineStateDesc.InputLayout;
			inputLayoutDesc.pInputElementDescs = dxVertexShader->m_InputElementDescs.data();
			inputLayoutDesc.NumElements = static_cast<UINT>(dxVertexShader->m_InputElementDescs.size());

			bool hasRenderTarget = targetInfo.renderTargetFormat != DXGI_FORMAT_UNKNOWN;
			bool hasDepthStencil = targetInfo.depthStencilFormat != DXGI_FORMAT_UNKNOWN;

			D3D12_BLEND_DESC& blendDesc = pipelineStateDesc.BlendState;
			blendDesc.AlphaToCoverageEnable = false;
			blendDesc.RenderTarget[0].BlendEnable = hasRenderTarget;
			blendDesc.RenderTarget[0].SrcBlend = GetBlend(passData.blendSrcColor);
			blendDesc.RenderTarget[0].DestBlend = GetBlend(passData.blendDstColor);
			blendDesc.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].SrcBlendAlpha = GetBlend(passData.blendSrcAlpha);
			blendDesc.RenderTarget[0].DestBlendAlpha = GetBlend(passData.blendDstAlpha);
			blendDesc.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].RenderTargetWriteMask = hasRenderTarget ? D3D12_COLOR_WRITE_ENABLE_ALL : 0;

			D3D12_RASTERIZER_DESC& rasterizerDesc = pipelineStateDesc.RasterizerState;
			rasterizerDesc.FillMode = isSolid ? D3D12_FILL_MODE_SOLID : D3D12_FILL_MODE_WIREFRAME;
			rasterizerDesc.CullMode = static_cast<D3D12_CULL_MODE>(static_cast<uint32_t>(passData.cullMode) + 1);;
			rasterizerDesc.FrontCounterClockwise = isCounterClockwise;
			rasterizerDesc.DepthBias = depthBias;
			rasterizerDesc.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
			rasterizerDesc.SlopeScaledDepthBias = slopeDepthBias;
			rasterizerDesc.MultisampleEnable = true;
			rasterizerDesc.AntialiasedLineEnable = true;
			rasterizerDesc.ForcedSampleCount = 0;
			rasterizerDesc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

			D3D12_DEPTH_STENCIL_DESC& depthStencilDesc = pipelineStateDesc.DepthStencilState;
			depthStencilDesc.DepthEnable = hasDepthStencil;
			depthStencilDesc.DepthWriteMask = passData.zWrite == ZWrite::On ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
			depthStencilDesc.DepthFunc = static_cast<D3D12_COMPARISON_FUNC>(static_cast<uint32_t>(passData.zTest) + 1);
			depthStencilDesc.StencilEnable = false;
			depthStencilDesc.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
			depthStencilDesc.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
			depthStencilDesc.BackFace = depthStencilDesc.FrontFace;

			m_Device->m_Device->CreateGraphicsPipelineState(&pipelineStateDesc, IID_PPV_ARGS(&pipelineState.pipelineState));
			m_PipelineStates.insert_or_assign(pipelineStateKey, pipelineState);
			renderState.pipelineState = pipelineState.pipelineState;
		}
		return renderState;
	}

	void GfxRenderStateCacheDX12::FillRenderState(Material* material, GfxRenderStateDX12& renderState, const GfxBindingStateDX12& bindingState)
	{
		for (auto& buffer : bindingState.vertexBuffers)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::s_PointerCache.Get(m_Device->m_BindedBuffers[buffer.bindingIndex].second);
			if (buffer.bufferSlot != UINT8_MAX)
			{
				renderState.vertexConstantBuffers[buffer.bufferSlot] = dxBuffer->m_ConstantBufferView.GetCPU();
				renderState.vertexConstantBuffersCount = std::max(renderState.vertexConstantBuffersCount, buffer.bufferSlot + 1u);
			}
			if (buffer.srvSlot != UINT8_MAX)
			{
				if (dxBuffer->m_State != SHADER_RESOURCE_STATE)
				{
					m_Device->TransitionBarrier(dxBuffer->m_Resource.Get(), dxBuffer->m_State, SHADER_RESOURCE_STATE);
					dxBuffer->m_State = SHADER_RESOURCE_STATE;
				}
				renderState.vertexShaderResourceViews[buffer.srvSlot] = dxBuffer->m_ShaderResourceView.GetCPU();
				renderState.vertexShaderResourceViewsCount = std::max(renderState.vertexShaderResourceViewsCount, buffer.srvSlot + 1u);
			}
		}

		for (auto& buffer : bindingState.geometryBuffers)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::s_PointerCache.Get(m_Device->m_BindedBuffers[buffer.bindingIndex].second);
			if (buffer.bufferSlot != UINT8_MAX)
			{
				renderState.geometryConstantBuffers[buffer.bufferSlot] = dxBuffer->m_ConstantBufferView.GetCPU();
				renderState.geometryConstantBuffersCount = std::max(renderState.geometryConstantBuffersCount, buffer.bufferSlot + 1u);
			}
			if (buffer.srvSlot != UINT8_MAX)
			{
				/*if (dxBuffer->m_State != SHADER_RESOURCE_STATE)
				{
					m_Device->TransitionBarrier(dxBuffer->m_Resource.Get(), dxBuffer->m_State, SHADER_RESOURCE_STATE);
					dxBuffer->m_State = SHADER_RESOURCE_STATE;
				}*/
				//renderState.geometryShaderResourceViews[buffer.srvSlot] = dxBuffer->m_ShaderResourceView.GetCPU();
				//renderState.geometryShaderResourceViewsCount = std::max(renderState.geometryShaderResourceViewsCount, buffer.srvSlot);
			}
		}

		for (auto& buffer : bindingState.pixelBuffers)
		{
			GfxBufferDX12* dxBuffer = GfxBufferDX12::s_PointerCache.Get(m_Device->m_BindedBuffers[buffer.bindingIndex].second);
			if (buffer.bufferSlot != UINT8_MAX)
			{
				renderState.pixelConstantBuffers[buffer.bufferSlot] = dxBuffer->m_ConstantBufferView.GetCPU();
				renderState.pixelConstantBuffersCount = std::max(renderState.pixelConstantBuffersCount, buffer.bufferSlot + 1u);
			}
			if (buffer.srvSlot != UINT8_MAX)
			{
				if (dxBuffer->m_State != SHADER_RESOURCE_STATE)
				{
					m_Device->TransitionBarrier(dxBuffer->m_Resource.Get(), dxBuffer->m_State, SHADER_RESOURCE_STATE);
					dxBuffer->m_State = SHADER_RESOURCE_STATE;
				}
				renderState.pixelShaderResourceViews[buffer.srvSlot] = dxBuffer->m_ShaderResourceView.GetCPU();
				renderState.pixelShaderResourceViewsCount = std::max(renderState.pixelShaderResourceViewsCount, buffer.srvSlot + 1u);
			}
		}

		for (auto& texture : bindingState.vertexTextures)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::s_PointerCache.Get(texture.isGlobal ? m_Device->m_BindedTextures[texture.bindingIndex].second : GetTextureIndex(material, texture.bindingIndex));
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			if (dxTexture->m_State != SHADER_RESOURCE_STATE)
			{
				m_Device->TransitionBarrier(dxTexture->m_Resource.Get(), dxTexture->m_State, SHADER_RESOURCE_STATE);
				dxTexture->m_State = SHADER_RESOURCE_STATE;
			}
			renderState.vertexShaderResourceViews[texture.srvSlot] = dxTexture->m_ShaderResourceView.GetCPU();
			renderState.vertexShaderResourceViewsCount = std::max(renderState.vertexShaderResourceViewsCount, texture.srvSlot + 1u);
			if (texture.samplerSlot != UINT8_MAX)
			{
				uint8_t sampler = dxTexture->m_Sampler;
				if (sampler == UINT8_MAX)
				{
					sampler = m_Device->GetSampler(dxTexture->m_WrapMode, dxTexture->m_FilterMode);
					dxTexture->m_Sampler = sampler;
				}
				renderState.vertexSamplers[texture.samplerSlot] = sampler;
				renderState.vertexSamplersCount = std::max(renderState.vertexSamplersCount, texture.samplerSlot + 1u);
			}
		}

		for (auto& texture : bindingState.pixelTextures)
		{
			GfxTextureDX12* dxTexture = GfxTextureDX12::s_PointerCache.Get(texture.isGlobal ? m_Device->m_BindedTextures[texture.bindingIndex].second : GetTextureIndex(material, texture.bindingIndex));
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			if (dxTexture->m_State != SHADER_RESOURCE_STATE)
			{
				m_Device->TransitionBarrier(dxTexture->m_Resource.Get(), dxTexture->m_State, SHADER_RESOURCE_STATE);
				dxTexture->m_State = SHADER_RESOURCE_STATE;
			}
			renderState.pixelShaderResourceViews[texture.srvSlot] = dxTexture->m_ShaderResourceView.GetCPU();
			renderState.pixelShaderResourceViewsCount = std::max(renderState.pixelShaderResourceViewsCount, texture.srvSlot + 1u);
			if (texture.samplerSlot != UINT8_MAX)
			{
				uint8_t sampler = dxTexture->m_Sampler;
				if (sampler == UINT8_MAX)
				{
					sampler = m_Device->GetSampler(dxTexture->m_WrapMode, dxTexture->m_FilterMode);
					dxTexture->m_Sampler = sampler;
				}
				renderState.pixelSamplers[texture.samplerSlot] = sampler;
				renderState.pixelSamplersCount = std::max(renderState.pixelSamplersCount, texture.samplerSlot + 1u);
			}
		}
	}
}