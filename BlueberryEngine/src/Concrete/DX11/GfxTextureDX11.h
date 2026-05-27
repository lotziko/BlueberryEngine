#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"
#include "..\..\Blueberry\Graphics\GfxPointerCache.h"

namespace Blueberry
{
	class GfxTextureDX11 : public GfxTexture
	{
	public:
		GfxTextureDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxTextureDX11();
		
		bool Initialize(const TextureProperties& properties);

		ID3D11Resource* GetTexture() const;
		ID3D11ShaderResourceView* GetSRV() const;
		ID3D11RenderTargetView* GetRTV() const;
		ID3D11RenderTargetView* GetRTV(uint32_t arraySlice, uint32_t mipSlice);

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
		bool Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, uint32_t subresourceCount, const TextureProperties& properties);

	private:
		ComPtr<ID3D11Resource> m_Texture;
		ComPtr<ID3D11ShaderResourceView> m_ShaderResourceView;
		ComPtr<ID3D11SamplerState> m_SamplerState;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;
		ComPtr<ID3D11DepthStencilView> m_DepthStencilView;
		ComPtr<ID3D11UnorderedAccessView> m_UnorderedAccessView;
		ComPtr<ID3D11Resource> m_StagingTexture;

		List<ComPtr<ID3D11RenderTargetView>> m_SlicesRenderTargetViews;

		DXGI_FORMAT m_Format = DXGI_FORMAT_UNKNOWN;
		TextureDimension m_Dimension = TextureDimension::Texture2D;
		uint32_t m_Width = 0;
		uint32_t m_Height = 0;
		uint32_t m_Depth = 0;
		uint32_t m_AntiAliasing = 1;
		uint32_t m_ArraySize = 0;
		uint32_t m_MipLevels = 1;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Bilinear;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		static GfxPointerCache<GfxTextureDX11> s_PointerCache;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
		friend class GfxComputeRenderStateCacheDX11;
	};
}