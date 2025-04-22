#include "bbpch.h"
#include "GfxRenderStateCacheDX11.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Concrete\DX11\GfxDeviceDX11.h"

#include "Blueberry\Graphics\GfxShader.h"
#include "Concrete\DX11\GfxShaderDX11.h"
#include "Concrete\DX11\GfxTextureDX11.h"
#include "Concrete\DX11\GfxBufferDX11.h"

namespace Blueberry
{
	bool GfxRenderStateKeyDX11::operator==(const GfxRenderStateKeyDX11& other) const
	{
		return std::memcmp(this, &other, sizeof(GfxRenderStateKeyDX11)) == 0;
	}

	bool GfxRenderStateKeyDX11::operator!=(const GfxRenderStateKeyDX11& other) const
	{
		return std::memcmp(this, &other, sizeof(GfxRenderStateKeyDX11)) != 0;
	}

	GfxRenderStateCacheDX11::GfxRenderStateCacheDX11(GfxDeviceDX11* device) : m_Device(device)
	{
	}

	const GfxRenderStateDX11 GfxRenderStateCacheDX11::GetState(Material* material, const uint8_t& passIndex)
	{
		uint64_t keywordMask = static_cast<uint64_t>(Shader::GetActiveKeywordsMask()) | (static_cast<uint64_t>(material->GetActiveKeywordsMask()) << 32);
		uint32_t crc = m_Device->GetCRC() ^ material->GetCRC();
		ObjectId objectId = material->GetObjectId();
		
		GfxRenderStateDX11 renderState;
		GfxGlobalBindingsStateDX11 globalBindingsState;
		GfxRenderStateKeyDX11 key = { keywordMask, objectId, passIndex };
		auto it = m_RenderStates.find(key);
		if (it != m_RenderStates.end() && it->second.first.materialCrc == crc)
		{
			renderState = it->second.first;
			globalBindingsState = it->second.second;
		}
		else
		{
			uint32_t size = m_RenderStates.size();
			renderState = {};
			globalBindingsState = {};
			GfxPassData passData = GetPassData(material, passIndex);
			renderState.isValid = passData.isValid;
			renderState.materialCrc = crc;

			if (!renderState.isValid)
			{
				return renderState;
			}

			auto dxVertexShader = static_cast<GfxVertexShaderDX11*>(passData.vertexShader);
			auto dxGeometryShader = static_cast<GfxGeometryShaderDX11*>(passData.geometryShader);
			auto dxFragmentShader = static_cast<GfxFragmentShaderDX11*>(passData.fragmentShader);

			renderState.dxVertexShader = dxVertexShader;
			renderState.vertexShader = dxVertexShader->m_Shader.Get();

			// Vertex global constant buffers
			for (auto it = dxVertexShader->m_ConstantBufferSlots.begin(); it != dxVertexShader->m_ConstantBufferSlots.end(); it++)
			{
				auto pair = m_Device->m_BindedConstantBuffers.find(it->first);
				if (pair != m_Device->m_BindedConstantBuffers.end())
				{
					renderState.vertexConstantBuffers[it->second] = pair->second->m_Buffer.Get();
				}
			}

			// Vertex global structured buffers
			for (auto it = dxVertexShader->m_StructuredBufferSlots.begin(); it != dxVertexShader->m_StructuredBufferSlots.end(); it++)
			{
				auto pair = m_Device->m_BindedStructuredBuffers.find(it->first);
				if (pair != m_Device->m_BindedStructuredBuffers.end())
				{
					uint32_t bufferSlotIndex = it->second.first;
					uint32_t shaderResourceViewSlotIndex = it->second.second;
					auto dxBuffer = pair->second;
					renderState.vertexShaderResourceViews[shaderResourceViewSlotIndex] = dxBuffer->m_ShaderResourceView.Get();
				}
			}

			if (dxGeometryShader != nullptr)
			{
				renderState.geometryShader = dxGeometryShader->m_Shader.Get();

				// Geometry global constant buffers
				for (auto it = dxGeometryShader->m_ConstantBufferSlots.begin(); it != dxGeometryShader->m_ConstantBufferSlots.end(); it++)
				{
					auto pair = m_Device->m_BindedConstantBuffers.find(it->first);
					if (pair != m_Device->m_BindedConstantBuffers.end())
					{
						renderState.geometryConstantBuffers[it->second] = pair->second->m_Buffer.Get();
					}
				}
			}
			renderState.pixelShader = dxFragmentShader->m_Shader.Get();

			// Fragment global constant buffers
			for (auto it = dxFragmentShader->m_ConstantBufferSlots.begin(); it != dxFragmentShader->m_ConstantBufferSlots.end(); it++)
			{
				auto pair = m_Device->m_BindedConstantBuffers.find(it->first);
				if (pair != m_Device->m_BindedConstantBuffers.end())
				{
					renderState.pixelConstantBuffers[it->second] = pair->second->m_Buffer.Get();
				}
			}

			// Fragment material textures
			for (auto it = dxFragmentShader->m_TextureSlots.begin(); it != dxFragmentShader->m_TextureSlots.end(); it++)
			{
				Texture* texture = material->GetTexture(it->first);
				if (texture != nullptr)
				{
					auto dxTexture = static_cast<GfxTextureDX11*>(texture->Get());
					renderState.pixelShaderResourceViews[it->second.first] = dxTexture->m_ResourceView.Get();
					if (it->second.second != -1)
					{
						renderState.pixelSamplerStates[it->second.second] = dxTexture->m_SamplerState.Get();
					}
				}
			}

			// Can replace this with an index to binded global textures
			// Fragment global textures
			for (auto it = dxFragmentShader->m_TextureSlots.begin(); it != dxFragmentShader->m_TextureSlots.end(); it++)
			{
				for (auto it1 = m_Device->m_BindedTextures.begin(); it1 < m_Device->m_BindedTextures.end(); ++it1)
				{
					if (it1->first == it->first)
					{
						uint8_t textureSlotIndex = it->second.first;
						uint8_t samplerSlotIndex = it->second.second;
						globalBindingsState.fragmentTextures[globalBindingsState.fragmentTextureCount] = { &(it1->second), textureSlotIndex, samplerSlotIndex };
						++globalBindingsState.fragmentTextureCount;
						break;
					}
				}
			}

			renderState.rasterizerState = m_Device->GetRasterizerState(passData.cullMode);
			renderState.depthStencilState = m_Device->GetDepthStencilState(passData.zTest, passData.zWrite);
			renderState.blendState = m_Device->GetBlendState(passData.blendSrcColor, passData.blendSrcAlpha, passData.blendDstColor, passData.blendDstAlpha);

			m_RenderStates.insert_or_assign(key, std::make_pair(renderState, globalBindingsState));
		}

		// Fragment global textures
		for (uint8_t i = 0; i < globalBindingsState.fragmentTextureCount; ++i)
		{
			GfxGlobalBindingsStateDX11::TextureData data = globalBindingsState.fragmentTextures[i];
			auto dxTexture = (*data.texture);
			if (dxTexture != nullptr)
			{
				renderState.pixelShaderResourceViews[data.srvSlot] = dxTexture->m_ResourceView.Get();
				if (data.samplerSlot != -1)
				{
					renderState.pixelSamplerStates[data.samplerSlot] = dxTexture->m_SamplerState.Get();
				}
			}
		}

		return renderState;
	}
}