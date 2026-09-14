#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\GfxRenderStateCache.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	struct GfxRenderStateKeyDX11
	{
		uint64_t keywordsMask; // global + material
		uint64_t passId;
		ObjectId materialId;
		uint32_t depthBias;
		float slopeDepthBias;
		bool isCounterClockwise;
		bool isSolid;
		uint16_t padding = 0;

		bool operator==(const GfxRenderStateKeyDX11& other) const;
		bool operator!=(const GfxRenderStateKeyDX11& other) const;
	};
}

template <>
struct std::hash<Blueberry::GfxRenderStateKeyDX11>
{
	size_t operator()(const Blueberry::GfxRenderStateKeyDX11& key) const
	{
		return std::hash<uint64_t>()(key.keywordsMask) ^ (std::hash<uint64_t>()(key.passId) << 1) ^ (std::hash<uint32_t>()(key.materialId) << 2);
	}
};

namespace Blueberry
{
	class Material;
	class GfxDeviceDX11;
	class GfxTextureDX11;
	class GfxVertexShaderDX11;
	class VertexLayout;

	// store current state in gfxDevice and compare it with new, and modify if they are different
	// also can do loop check for samplers, SRV and buffers
	struct GfxRenderStateDX11
	{
		GfxVertexShaderDX11* dxVertexShader;

		ID3D11InputLayout* inputLayout;
		ID3D11VertexShader* vertexShader;
		ID3D11GeometryShader* geometryShader;
		ID3D11PixelShader* pixelShader;

		ID3D11ShaderResourceView* vertexShaderResourceViews[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 4];
		ID3D11SamplerState* vertexSamplerStates[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
		ID3D11ShaderResourceView* pixelShaderResourceViews[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 4];
		ID3D11SamplerState* pixelSamplerStates[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];

		ID3D11Buffer* vertexConstantBuffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
		ID3D11Buffer* geometryConstantBuffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
		ID3D11Buffer* pixelConstantBuffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];

		ID3D11RasterizerState* rasterizerState;
		ID3D11DepthStencilState* depthStencilState;
		ID3D11BlendState* blendState;

		bool isValid;
		uint32_t crc;
	};

	struct GfxTextureBindingDX11
	{
		uint32_t bindingIndex;
		bool isGlobal;
		uint8_t srvSlot;
		uint8_t samplerSlot;
	};

	struct GfxBufferBindingDX11
	{
		uint32_t bindingIndex;
		bool isGlobal;
		uint8_t bufferSlot;
		uint8_t srvSlot;
	};

	struct GfxStaticSamplerBindingDX11
	{
		ID3D11SamplerState* samplerState;
		uint8_t slotIndex;
	};

	struct GfxBindingStateDX11
	{
		List<GfxTextureBindingDX11> vertexTextures;
		List<GfxTextureBindingDX11> pixelTextures;

		List<GfxBufferBindingDX11> vertexBuffers;
		List<GfxBufferBindingDX11> geometryBuffers;
		List<GfxBufferBindingDX11> pixelBuffers;

		List<GfxStaticSamplerBindingDX11> vertexStaticSamplers;
		List<GfxStaticSamplerBindingDX11> pixelStaticSamplers;
	};

	class GfxRenderStateCacheDX11 : public GfxRenderStateCache
	{
	public:
		GfxRenderStateCacheDX11() = default;
		GfxRenderStateCacheDX11(GfxDeviceDX11* device);
		~GfxRenderStateCacheDX11();

		const GfxRenderStateDX11 GetState(Material* material, uint64_t passId, VertexLayout* meshLayout, uint32_t depthBias, float slopeDepthBias, bool isCounterClockwise, bool isSolid);
		
	private:
		void FillRenderState(Material* material, GfxRenderStateDX11& renderState, const GfxBindingStateDX11& bindingState);
		ID3D11InputLayout* GetLayout(GfxVertexShaderDX11* shader, VertexLayout* meshLayout);

	private:
		GfxDeviceDX11* m_Device;
		Dictionary<GfxRenderStateKeyDX11, std::pair<GfxRenderStateDX11, GfxBindingStateDX11>> m_RenderStates;
		Dictionary<size_t, ID3D11InputLayout*> m_InputLayouts;
	};
}