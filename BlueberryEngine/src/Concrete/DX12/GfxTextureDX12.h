#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"
#include "GfxDescriptorHeapDX12.h"
#include "..\..\Blueberry\Graphics\GfxPointerCache.h"

namespace Blueberry
{
	class GfxDeviceDX12;

	class GfxTextureDX12 : public GfxTexture
	{
	public:
		GfxTextureDX12(GfxDeviceDX12* device);
		virtual ~GfxTextureDX12();

		bool Initialize(const TextureProperties& properties);

		ID3D12Resource* GetResource() const;
		const GfxHandleDX12& GetSRV() const;
		const GfxHandleDX12& GetRTV() const;
		const GfxHandleDX12& GetRTV(uint32_t arraySlice, uint32_t mipSlice);
		D3D12_RESOURCE_STATES GetState() const;

		virtual uint32_t GetWidth() const override;
		virtual uint32_t GetHeight() const override;
		virtual TextureFormat GetFormat() const override;
		virtual void* GetHandle() override;

		virtual void GetData(void* data, const Rectangle& area) override;
		virtual void GetData(void* data) override;
		virtual void SetData(void* data, size_t size) override;

		virtual void SetWrapMode(WrapMode wrapMode) override;
		virtual void SetFilterMode(FilterMode filterMode) override;
		virtual void SetName(const String& name) override;

	private:
		void GatherSubresources(const void* data, List<D3D12_SUBRESOURCE_DATA>& subresourceDatas);
		bool Initialize(D3D12_SUBRESOURCE_DATA* subresourceData, uint32_t subresourceCount, const TextureProperties& properties);

	private:
		ComPtr<ID3D12Resource> m_Resource;
		GfxHandleDX12 m_ShaderResourceView;
		GfxHandleDX12 m_RenderTargetView;
		GfxHandleDX12 m_DepthStencilView;
		GfxHandleDX12 m_UnorderedAccessView;
		uint8_t m_Sampler = UINT8_MAX;

		GfxRingHandleDX12 m_HandleShaderResourceView;
		uint64_t m_HandleGeneration = 0;

		List<GfxHandleDX12> m_SlicesRenderTargetViews;

		DXGI_FORMAT m_Format = DXGI_FORMAT_UNKNOWN;
		uint32_t m_Width = 0;
		uint32_t m_Height = 0;
		uint32_t m_Depth = 0;
		uint32_t m_AntiAliasing = 1;
		uint32_t m_Quality = 0;
		uint32_t m_ArraySize = 0;
		uint32_t m_MipLevels = 1;
		TextureDimension m_Dimension = TextureDimension::Texture2D;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Bilinear;
		D3D12_RESOURCE_STATES m_State = D3D12_RESOURCE_STATE_COMMON;

		ID3D12Device* m_Device;
		GfxDeviceDX12* m_GfxDevice;

		static GfxPointerCache<GfxTextureDX12> s_PointerCache;

		friend class GfxDeviceDX12;
		friend class GfxRenderStateCacheDX12;
		friend class GfxComputeRenderStateCacheDX12;
	};
}