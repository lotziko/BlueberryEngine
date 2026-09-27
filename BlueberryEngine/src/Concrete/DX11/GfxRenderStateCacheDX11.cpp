#include "GfxRenderStateCacheDX11.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\VertexLayout.h"

#include "GfxDeviceDX11.h"
#include "GfxShaderDX11.h"
#include "GfxTextureDX11.h"
#include "GfxBufferDX11.h"

namespace Blueberry
{
	bool GfxRenderStateKeyDX11::operator==(const GfxRenderStateKeyDX11& other) const
	{
		return memcmp(this, &other, sizeof(GfxRenderStateKeyDX11)) == 0;
	}

	bool GfxRenderStateKeyDX11::operator!=(const GfxRenderStateKeyDX11& other) const
	{
		return memcmp(this, &other, sizeof(GfxRenderStateKeyDX11)) != 0;
	}

	GfxRenderStateCacheDX11::GfxRenderStateCacheDX11(GfxDeviceDX11* device) : m_Device(device)
	{
		m_RenderStates.reserve(4096);
	}

	GfxRenderStateCacheDX11::~GfxRenderStateCacheDX11()
	{
		for (auto& pair : m_InputLayouts)
		{
			pair.second->Release();
		}
		m_InputLayouts.clear();
	}

	const GfxRenderStateDX11 GfxRenderStateCacheDX11::GetState(Material* material, uint64_t passId, VertexLayout* meshLayout, uint32_t depthBias, float slopeDepthBias, bool isCounterClockwise, bool isSolid)
	{
		uint64_t keywordMask = static_cast<uint64_t>(Shader::GetActiveKeywordsMask()) | (static_cast<uint64_t>(material->GetActiveKeywordsMask()) << 32);
		ObjectId objectId = material->GetObjectId(); // Maybe also use shader id to be able to switch it
		uint32_t crc = material->GetCRC();

		GfxRenderStateDX11 renderState;
		GfxRenderStateKeyDX11 key = { keywordMask, passId, objectId, depthBias, slopeDepthBias, isCounterClockwise, isSolid };
		auto it = m_RenderStates.find(key);
		if (it != m_RenderStates.end() && crc == it->second.first.crc)
		{
			renderState = it->second.first;
			FillRenderState(material, renderState, it->second.second);
		}
		else
		{
			uint32_t size = static_cast<uint32_t>(m_RenderStates.size());
			if (size > 4096)
			{
				m_RenderStates.clear();
			}
			renderState = {};
			GfxPassData passData = GetPassData(material, passId);
			renderState.isValid = passData.isValid;
			renderState.crc = crc;

			if (!renderState.isValid)
			{
				return renderState;
			}

			auto dxVertexShader = static_cast<GfxVertexShaderDX11*>(passData.vertexShader);
			auto dxGeometryShader = static_cast<GfxGeometryShaderDX11*>(passData.geometryShader);
			auto dxFragmentShader = static_cast<GfxFragmentShaderDX11*>(passData.fragmentShader);

			renderState.dxVertexShader = dxVertexShader;
			renderState.vertexShader = dxVertexShader->m_Shader.Get();

			GfxBindingStateDX11 bindingState = {};
			List<size_t> usedTextures = {};

			// Vertex global constant buffers
			for (auto it = dxVertexShader->m_ConstantBufferSlots.begin(); it != dxVertexShader->m_ConstantBufferSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
				{
					if (it1->id == it->first)
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
					if (it1->id == it->first)
					{
						bindingState.vertexBuffers.push_back({ offset, true, UINT8_MAX, it->second });
						break;
					}
				}
			}

			// Vertex material textures
			for (auto it = dxVertexShader->m_TextureSRVSamplerSlots.begin(); it != dxVertexShader->m_TextureSRVSamplerSlots.end(); it++)
			{
				uint32_t offset = GetTextureSlot(material, it->first);
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
					if (it1->id == it->first)
					{
						if (std::find(usedTextures.begin(), usedTextures.end(), it1->id) == usedTextures.end())
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
				renderState.geometryShader = dxGeometryShader->m_Shader.Get();

				// Geometry global constant buffers
				for (auto it = dxGeometryShader->m_ConstantBufferSlots.begin(); it != dxGeometryShader->m_ConstantBufferSlots.end(); it++)
				{
					uint32_t offset = 0;
					for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
					{
						if (it1->id == it->first)
						{
							bindingState.geometryBuffers.push_back({ offset, true, it->second, UINT8_MAX });
							break;
						}
					}
				}
			}
			renderState.pixelShader = dxFragmentShader->m_Shader.Get();

			// Fragment global constant buffers
			for (auto it = dxFragmentShader->m_ConstantBufferSlots.begin(); it != dxFragmentShader->m_ConstantBufferSlots.end(); it++)
			{
				uint32_t offset = 0;
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1, ++offset)
				{
					if (it1->id == it->first)
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
					if (it1->id == it->first)
					{
						bindingState.pixelBuffers.push_back({ offset, true, UINT8_MAX, it->second });
						break;
					}
				}
			}

			// Fragment material textures
			for (auto it = dxFragmentShader->m_TextureSRVSamplerSlots.begin(); it != dxFragmentShader->m_TextureSRVSamplerSlots.end(); it++)
			{
				uint32_t offset = GetTextureSlot(material, it->first);
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
					if (it1->id == it->first)
					{
						if (std::find(usedTextures.begin(), usedTextures.end(), it1->id) == usedTextures.end())
						{
							bindingState.pixelTextures.push_back({ offset, true, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
						}
						break;
					}
				}
			}

			// Vertex static samplers
			for (size_t i = 0; i < dxVertexShader->m_StaticSamplerSlots.size(); ++i)
			{
				auto& samplerSlot = dxVertexShader->m_StaticSamplerSlots[i];
				bindingState.vertexStaticSamplers.push_back({ m_Device->GetSamplerState(std::get<1>(samplerSlot), std::get<0>(samplerSlot)), static_cast<uint8_t>(std::get<2>(samplerSlot)) });
			}

			// Fragment static samplers
			for (size_t i = 0; i < dxFragmentShader->m_StaticSamplerSlots.size(); ++i)
			{
				auto& samplerSlot = dxFragmentShader->m_StaticSamplerSlots[i];
				bindingState.pixelStaticSamplers.push_back({ m_Device->GetSamplerState(std::get<1>(samplerSlot), std::get<0>(samplerSlot)), static_cast<uint8_t>(std::get<2>(samplerSlot)) });
			}

			renderState.rasterizerState = m_Device->GetRasterizerState(passData.cullMode, depthBias, slopeDepthBias, isCounterClockwise, isSolid);
			renderState.depthStencilState = m_Device->GetDepthStencilState(passData.zTest, passData.zWrite);
			renderState.blendState = m_Device->GetBlendState(passData.blendSrcColor, passData.blendSrcAlpha, passData.blendDstColor, passData.blendDstAlpha);
			
			m_RenderStates.insert_or_assign(key, std::make_pair(renderState, bindingState));
			FillRenderState(material, renderState, bindingState);
		}
		renderState.inputLayout = GetLayout(renderState.dxVertexShader, meshLayout);
		return renderState;
	}

	void GfxRenderStateCacheDX11::FillRenderState(Material* material, GfxRenderStateDX11& renderState, const GfxBindingStateDX11& bindingState)
	{
		for (auto& buffer : bindingState.vertexBuffers)
		{
			GfxBufferDX11* dxBuffer = GfxBufferDX11::Get(m_Device->m_BindedBuffers[buffer.bindingIndex].index);
			if (buffer.bufferSlot != UINT8_MAX)
			{
				renderState.vertexConstantBuffers[buffer.bufferSlot] = dxBuffer->GetBuffer();
			}
			if (buffer.srvSlot != UINT8_MAX)
			{
				renderState.vertexShaderResourceViews[buffer.srvSlot] = dxBuffer->GetShaderResourceView();
			}
		}

		if (renderState.geometryShader != nullptr)
		{
			for (auto& buffer : bindingState.geometryBuffers)
			{
				GfxBufferDX11* dxBuffer = GfxBufferDX11::Get(m_Device->m_BindedBuffers[buffer.bindingIndex].index);
				if (buffer.bufferSlot != UINT8_MAX)
				{
					renderState.geometryConstantBuffers[buffer.bufferSlot] = dxBuffer->GetBuffer();
				}
				if (buffer.srvSlot != UINT8_MAX)
				{
					//renderState.geometryShaderResourceViews[buffer.srvSlot] = dxBuffer->GetShaderResourceView();
				}
			}
		}

		for (auto& buffer : bindingState.pixelBuffers)
		{
			GfxBufferDX11* dxBuffer = GfxBufferDX11::Get(m_Device->m_BindedBuffers[buffer.bindingIndex].index);
			if (buffer.bufferSlot != UINT8_MAX)
			{
				renderState.pixelConstantBuffers[buffer.bufferSlot] = dxBuffer->GetBuffer();
			}
			if (buffer.srvSlot != UINT8_MAX)
			{
				renderState.pixelShaderResourceViews[buffer.srvSlot] = dxBuffer->GetShaderResourceView();
			}
		}

		for (auto& texture : bindingState.vertexTextures)
		{
			uint32_t mip;
			uint32_t index;
			if (texture.isGlobal)
			{
				auto& bindedTexture = m_Device->m_BindedTextures[texture.bindingIndex];
				mip = bindedTexture.mip;
				index = bindedTexture.index;
			}
			else
			{
				mip = UINT32_MAX;
				index = GetTextureIndex(material, texture.bindingIndex);
			}
			GfxTextureDX11* dxTexture = GfxTextureDX11::Get(index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			renderState.vertexShaderResourceViews[texture.srvSlot] = mip == UINT32_MAX ? dxTexture->GetShaderResourceView() : dxTexture->GetShaderResourceView(0, mip);
			if (texture.samplerSlot != UINT8_MAX)
			{
				ID3D11SamplerState* samplerState = dxTexture->GetSamplerState();
				if (samplerState == nullptr)
				{
					samplerState = m_Device->GetSamplerState(dxTexture->GetWrapMode(), dxTexture->GetFilterMode());
					dxTexture->SetSamplerState(samplerState);
				}
				renderState.vertexSamplerStates[texture.samplerSlot] = samplerState;
			}
		}

		for (auto& texture : bindingState.pixelTextures)
		{
			uint32_t mip;
			uint32_t index;
			if (texture.isGlobal)
			{
				auto& bindedTexture = m_Device->m_BindedTextures[texture.bindingIndex];
				mip = bindedTexture.mip;
				index = bindedTexture.index;
			}
			else
			{
				mip = UINT32_MAX;
				index = GetTextureIndex(material, texture.bindingIndex);
			}
			GfxTextureDX11* dxTexture = GfxTextureDX11::Get(index);
			if (dxTexture == nullptr)
			{
				BB_ERROR("Texture is missing.");
				continue;
			}
			renderState.pixelShaderResourceViews[texture.srvSlot] = mip == UINT32_MAX ? dxTexture->GetShaderResourceView() : dxTexture->GetShaderResourceView(0, mip);
			if (texture.samplerSlot != UINT8_MAX)
			{
				ID3D11SamplerState* samplerState = dxTexture->GetSamplerState();
				if (samplerState == nullptr)
				{
					samplerState = m_Device->GetSamplerState(dxTexture->GetWrapMode(), dxTexture->GetFilterMode());
					dxTexture->SetSamplerState(samplerState);
				}
				renderState.pixelSamplerStates[texture.samplerSlot] = samplerState;
			}
		}

		for (auto& binding : bindingState.vertexStaticSamplers)
		{
			renderState.vertexSamplerStates[binding.slotIndex] = binding.samplerState;
		}

		for (auto& binding : bindingState.pixelStaticSamplers)
		{
			renderState.pixelSamplerStates[binding.slotIndex] = binding.samplerState;
		}
	}

	ID3D11InputLayout* GfxRenderStateCacheDX11::GetLayout(GfxVertexShaderDX11* shader, VertexLayout* meshLayout)
	{
		size_t key = static_cast<uint64_t>(shader->m_Crc) | (static_cast<uint64_t>(meshLayout->GetCrc()) << 32);
		auto it = m_InputLayouts.find(key);
		if (it != m_InputLayouts.end())
		{
			return it->second;
		}
		else
		{
			for (uint32_t i = 0; i < RENDERABLE_VERTEX_ATTRIBUTE_COUNT; ++i)
			{
				uint32_t offset = meshLayout->GetOffset(i);
				uint8_t index = shader->m_LayoutIndices[i];
				if (index != UINT8_MAX)
				{
					shader->m_InputElementDescs[index].AlignedByteOffset = offset;
				}
			}
			ID3D11InputLayout* layout = shader->CreateLayout();
			m_InputLayouts.insert_or_assign(key, layout);
			return layout;
		}
	}
}