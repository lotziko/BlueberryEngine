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
		return keywordsMask == other.keywordsMask && materialId == other.materialId && passIndex == other.passIndex && crc == other.crc;
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
		ObjectId objectId = material->GetObjectId();
		
		GfxRenderStateDX11 renderState;
		GfxRenderStateKeyDX11 key = { keywordMask, objectId, passIndex, static_cast<uint64_t>(m_Device->GetCRC()) | (static_cast<uint64_t>(material->GetCRC()) << 32) };
		auto it = m_RenderStates.find(key);
		if (it != m_RenderStates.end())
		{
			renderState = it->second;
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
			BB_INFO(size);

			auto dxVertexShader = static_cast<GfxVertexShaderDX11*>(passData.vertexShader);
			auto dxGeometryShader = static_cast<GfxGeometryShaderDX11*>(passData.geometryShader);
			auto dxFragmentShader = static_cast<GfxFragmentShaderDX11*>(passData.fragmentShader);

			renderState.dxVertexShader = dxVertexShader;
			renderState.vertexShader = dxVertexShader->m_Shader.Get();

			// Vertex global constant buffers
			for (auto it = dxVertexShader->m_ConstantBufferSlots.begin(); it != dxVertexShader->m_ConstantBufferSlots.end(); it++)
			{
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1)
				{
					if (it1->first == it->first)
					{
						renderState.vertexConstantBuffers[it->second] = it1->second->m_Buffer.Get();
					}
				}
			}

			// Vertex global structured buffers
			for (auto it = dxVertexShader->m_StructuredBufferSlots.begin(); it != dxVertexShader->m_StructuredBufferSlots.end(); it++)
			{
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1)
				{
					if (it1->first == it->first)
					{
						uint32_t bufferSlotIndex = it->second.first;
						uint32_t shaderResourceViewSlotIndex = it->second.second;
						auto dxBuffer = it1->second;
						renderState.vertexShaderResourceViews[shaderResourceViewSlotIndex] = dxBuffer->m_ShaderResourceView.Get();
					}
				}
			}

			if (dxGeometryShader != nullptr)
			{
				renderState.geometryShader = dxGeometryShader->m_Shader.Get();

				// Geometry global constant buffers
				for (auto it = dxGeometryShader->m_ConstantBufferSlots.begin(); it != dxGeometryShader->m_ConstantBufferSlots.end(); it++)
				{
					for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1)
					{
						if (it1->first == it->first)
						{
							renderState.geometryConstantBuffers[it->second] = it1->second->m_Buffer.Get();
						}
					}
				}
			}
			renderState.pixelShader = dxFragmentShader->m_Shader.Get();

			// Fragment global constant buffers
			for (auto it = dxFragmentShader->m_ConstantBufferSlots.begin(); it != dxFragmentShader->m_ConstantBufferSlots.end(); it++)
			{
				for (auto it1 = m_Device->m_BindedBuffers.begin(); it1 < m_Device->m_BindedBuffers.end(); ++it1)
				{
					if (it1->first == it->first)
					{
						renderState.pixelConstantBuffers[it->second] = it1->second->m_Buffer.Get();
					}
				}
			}

			// Fragment material textures
			for (auto it = dxFragmentShader->m_TextureSlots.begin(); it != dxFragmentShader->m_TextureSlots.end(); it++)
			{
				Texture* texture = material->GetTexture(it->first);
				if (texture != nullptr && texture->GetState() == ObjectState::Default)
				{
					auto dxTexture = static_cast<GfxTextureDX11*>(texture->Get());
					renderState.pixelShaderResourceViews[it->second.first] = dxTexture->m_ResourceView.Get();
					if (it->second.second != 255)
					{
						renderState.pixelSamplerStates[it->second.second] = dxTexture->m_SamplerState.Get();
					}
				}
			}

			// Fragment global textures
			// Has a possiblity to collide with material textures if has same name
			for (auto it = dxFragmentShader->m_TextureSlots.begin(); it != dxFragmentShader->m_TextureSlots.end(); it++)
			{
				for (auto it1 = m_Device->m_BindedTextures.begin(); it1 < m_Device->m_BindedTextures.end(); ++it1)
				{
					if (it1->first == it->first)
					{
						renderState.pixelShaderResourceViews[it->second.first] = it1->second->m_ResourceView.Get();
						if (it->second.second != 255)
						{
							renderState.pixelSamplerStates[it->second.second] = it1->second->m_SamplerState.Get();
						}
						break;
					}
				}
			}

			renderState.rasterizerState = m_Device->GetRasterizerState(passData.cullMode);
			renderState.depthStencilState = m_Device->GetDepthStencilState(passData.zTest, passData.zWrite);
			renderState.blendState = m_Device->GetBlendState(passData.blendSrcColor, passData.blendSrcAlpha, passData.blendDstColor, passData.blendDstAlpha);

			m_RenderStates.insert_or_assign(key, renderState);
		}
		return renderState;
	}
}