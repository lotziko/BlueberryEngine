#pragma once

#include "Blueberry\Graphics\GfxRenderStateCache.h"

namespace Blueberry
{
	struct GfxRenderStateKeyDX11
	{
		uint64_t keywordsMask; // global + material
		ObjectId materialId;
		uint8_t passIndex;

		bool operator==(const GfxRenderStateKeyDX11& other) const;
		bool operator!=(const GfxRenderStateKeyDX11& other) const;
	};
}

template <>
struct std::hash<Blueberry::GfxRenderStateKeyDX11>
{
	size_t operator()(const Blueberry::GfxRenderStateKeyDX11& key) const
	{
		return std::hash<uint64_t>()(key.keywordsMask) ^ (std::hash<uint32_t>()(key.materialId) << 1) ^ (std::hash<uint8_t>()(key.passIndex) << 2);
	}
};

namespace Blueberry
{
	class Material;
	class GfxDeviceDX11;
	class GfxTextureDX11;

	// store current state in gfxDevice and compare it with new, and modify if they are different
	// also can do loop check for samplers, SRV and buffers
	struct GfxRenderStateDX11
	{
		ID3D11InputLayout* inputLayout;
		ID3D11VertexShader* vertexShader;
		ID3D11GeometryShader* geometryShader;
		ID3D11PixelShader* pixelShader;

		ID3D11ShaderResourceView* vertexShaderResourceViews[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 8];
		ID3D11SamplerState* vertexSamplerStates[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
		ID3D11ShaderResourceView* pixelShaderResourceViews[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 8];
		ID3D11SamplerState* pixelSamplerStates[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];

		ID3D11Buffer* vertexConstantBuffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
		ID3D11Buffer* geometryConstantBuffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
		ID3D11Buffer* pixelConstantBuffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];

		ID3D11RasterizerState* rasterizerState;
		ID3D11DepthStencilState* depthStencilState;
		ID3D11BlendState* blendState;

		bool isValid;
		uint32_t materialCrc;
	};

	struct GfxGlobalBindingsStateDX11
	{
		struct TextureData
		{
			GfxTextureDX11** texture;
			uint8_t srvSlot;
			uint8_t samplerSlot;
		};

		TextureData vertexTextures[16];
		TextureData fragmentTextures[16];
		uint8_t vertexTextureCount;
		uint8_t fragmentTextureCount;
	};

	class GfxRenderStateCacheDX11 : public GfxRenderStateCache
	{
	public:
		GfxRenderStateCacheDX11() = default;
		GfxRenderStateCacheDX11(GfxDeviceDX11* device);

		const GfxRenderStateDX11 GetState(Material* material, const uint8_t& passIndex);

		ska::flat_hash_map<GfxRenderStateKeyDX11, std::pair<GfxRenderStateDX11, GfxGlobalBindingsStateDX11>> m_RenderStates;

	private:
		GfxDeviceDX11* m_Device;
	};
}