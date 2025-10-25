#include "GfxRenderStateCacheDX11.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "..\DX11\GfxDeviceDX11.h"

#include "..\..\Blueberry\Graphics\GfxShader.h"
#include "..\DX11\GfxShaderDX11.h"
#include "..\DX11\GfxTextureDX11.h"
#include "..\DX11\GfxBufferDX11.h"

namespace Blueberry
{
	bool GfxRenderStateKeyDX11::operator==(const GfxRenderStateKeyDX11& other) const
	{
		return keywordsMask == other.keywordsMask && materialId == other.materialId && passIndex == other.passIndex;
	}

	bool GfxRenderStateKeyDX11::operator!=(const GfxRenderStateKeyDX11& other) const
	{
		return !(*this == other);
	}

	GfxRenderStateCacheDX11::GfxRenderStateCacheDX11(GfxDeviceDX11* device) : m_Device(device)
	{
		m_RenderStates.reserve(4096);
	}

	const GfxRenderStateDX11 GfxRenderStateCacheDX11::GetState(Material* material, const uint8_t& passIndex)
	{
		uint64_t keywordMask = static_cast<uint64_t>(Shader::GetActiveKeywordsMask()) | (static_cast<uint64_t>(material->GetActiveKeywordsMask()) << 32);
		ObjectId objectId = material->GetObjectId(); // Maybe also use shader id to be able to switch it
		
		GfxRenderStateDX11 renderState;
		GfxRenderStateKeyDX11 key = { keywordMask, objectId, passIndex };
		auto it = m_RenderStates.find(key);
		if (it != m_RenderStates.end())
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
			GfxPassData passData = GetPassData(material, passIndex);
			renderState.isValid = passData.isValid;

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
					if (it1->first == it->first)
					{
						bindingState.vertexBuffers.push_back({ offset, true, it->second, UINT8_MAX });
						break;
					}
				}
			}

			// Vertex global structured buffers
			for (auto it = dxVertexShader->m_StructuredBufferSlots.begin(); it != dxVertexShader->m_StructuredBufferSlots.end(); it++)
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
			for (auto it = dxVertexShader->m_TextureSlots.begin(); it != dxVertexShader->m_TextureSlots.end(); it++)
			{
				uint32_t offset = GetTextureIndex(material, it->first);
				if (offset != UINT32_MAX)
				{
					usedTextures.push_back(it->first);
					bindingState.vertexTextures.push_back({ offset, false, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
				}
			}

			// Vertex global textures
			for (auto it = dxVertexShader->m_TextureSlots.begin(); it != dxVertexShader->m_TextureSlots.end(); it++)
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
				renderState.geometryShader = dxGeometryShader->m_Shader.Get();

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
			renderState.pixelShader = dxFragmentShader->m_Shader.Get();

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
			for (auto it = dxFragmentShader->m_StructuredBufferSlots.begin(); it != dxFragmentShader->m_StructuredBufferSlots.end(); it++)
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
			for (auto it = dxFragmentShader->m_TextureSlots.begin(); it != dxFragmentShader->m_TextureSlots.end(); it++)
			{
				uint32_t offset = GetTextureIndex(material, it->first);
				if (offset != UINT32_MAX)
				{
					usedTextures.push_back(it->first);
					bindingState.pixelTextures.push_back({ offset, false, it->second.first, it->second.second != 255 ? it->second.second : UINT8_MAX });
				}
			}

			// Fragment global textures
			for (auto it = dxFragmentShader->m_TextureSlots.begin(); it != dxFragmentShader->m_TextureSlots.end(); it++)
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

			renderState.rasterizerState = m_Device->GetRasterizerState(passData.cullMode);
			renderState.depthStencilState = m_Device->GetDepthStencilState(passData.zTest, passData.zWrite);
			renderState.blendState = m_Device->GetBlendState(passData.blendSrcColor, passData.blendSrcAlpha, passData.blendDstColor, passData.blendDstAlpha);
			
			m_RenderStates.insert_or_assign(key, std::make_pair(renderState, bindingState));
			FillRenderState(material, renderState, bindingState);
		}
		return renderState;
	}

	void GfxRenderStateCacheDX11::FillRenderState(Material* material, GfxRenderStateDX11& renderState, const GfxBindingStateDX11& bindingState)
	{
		for (auto& buffer : bindingState.vertexBuffers)
		{
			GfxBufferDX11* dxBuffer = GfxBufferDX11::s_PointerCache.Get(m_Device->m_BindedBuffers[buffer.bindingIndex].second);
			if (buffer.bufferSlot != UINT8_MAX)
			{
				renderState.vertexConstantBuffers[buffer.bufferSlot] = dxBuffer->m_Buffer.Get();
			}
			if (buffer.srvSlot != UINT8_MAX)
			{
				renderState.vertexShaderResourceViews[buffer.srvSlot] = dxBuffer->m_ShaderResourceView.Get();
			}
		}

		if (renderState.geometryShader != nullptr)
		{
			for (auto& buffer : bindingState.geometryBuffers)
			{
				GfxBufferDX11* dxBuffer = GfxBufferDX11::s_PointerCache.Get(m_Device->m_BindedBuffers[buffer.bindingIndex].second);
				if (buffer.bufferSlot != UINT8_MAX)
				{
					renderState.geometryConstantBuffers[buffer.bufferSlot] = dxBuffer->m_Buffer.Get();
				}
				if (buffer.srvSlot != UINT8_MAX)
				{
					//renderState.geometryShaderResourceViews[buffer.srvSlot] = dxBuffer->m_ShaderResourceView.Get();
				}
			}
		}

		for (auto& buffer : bindingState.pixelBuffers)
		{
			GfxBufferDX11* dxBuffer = GfxBufferDX11::s_PointerCache.Get(m_Device->m_BindedBuffers[buffer.bindingIndex].second);
			if (buffer.bufferSlot != UINT8_MAX)
			{
				renderState.pixelConstantBuffers[buffer.bufferSlot] = dxBuffer->m_Buffer.Get();
			}
			if (buffer.srvSlot != UINT8_MAX)
			{
				renderState.pixelShaderResourceViews[buffer.srvSlot] = dxBuffer->m_ShaderResourceView.Get();
			}
		}

		for (auto& texture : bindingState.vertexTextures)
		{
			GfxTextureDX11* dxTexture = GfxTextureDX11::s_PointerCache.Get(texture.isGlobal ? m_Device->m_BindedTextures[texture.bindingIndex].second : GetTextureIndex(material, texture.bindingIndex));
			renderState.vertexShaderResourceViews[texture.srvSlot] = dxTexture->m_ShaderResourceView.Get();
			if (texture.samplerSlot != UINT8_MAX)
			{
				renderState.vertexSamplerStates[texture.samplerSlot] = dxTexture->m_SamplerState.Get();
			}
		}

		for (auto& texture : bindingState.pixelTextures)
		{
			GfxTextureDX11* dxTexture = GfxTextureDX11::s_PointerCache.Get(texture.isGlobal ? m_Device->m_BindedTextures[texture.bindingIndex].second : GetTextureIndex(material, texture.bindingIndex));
			renderState.pixelShaderResourceViews[texture.srvSlot] = dxTexture->m_ShaderResourceView.Get();
			if (texture.samplerSlot != UINT8_MAX)
			{
				renderState.pixelSamplerStates[texture.samplerSlot] = dxTexture->m_SamplerState.Get();
			}
		}
	}
}