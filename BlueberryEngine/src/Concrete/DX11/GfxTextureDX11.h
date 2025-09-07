#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	class GfxTextureDX11 : public GfxTexture
	{
	public:
		GfxTextureDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxTextureDX11() = default;
		
		bool Initialize(const TextureProperties& properties);

		ID3D11Resource* GetTexture() const;
		ID3D11ShaderResourceView* GetSRV() const;
		ID3D11RenderTargetView* GetRTV() const;
		ID3D11RenderTargetView* GetRTV(const uint32_t& slice);

		virtual uint32_t GetWidth() const override;
		virtual uint32_t GetHeight() const override;
		virtual void* GetHandle() override;

		virtual void GetData(void* target, const Rectangle& area) override;
		virtual void GetData(void* target) override;
		virtual void SetData(void* data, const size_t& size) override;

		virtual void GenerateMipMaps() override;

	private:
		uint32_t GetQualityLevel(const DXGI_FORMAT& format, const uint32_t& antiAliasing);
		bool Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, const uint32_t& subresourceCount, const TextureProperties& properties);

	private:
		ComPtr<ID3D11Resource> m_Texture;
		ComPtr<ID3D11ShaderResourceView> m_ResourceView;
		ComPtr<ID3D11SamplerState> m_SamplerState;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;
		ComPtr<ID3D11DepthStencilView> m_DepthStencilView;
		ComPtr<ID3D11UnorderedAccessView> m_UnorderedAccessView;
		ComPtr<ID3D11Resource> m_StagingTexture;

		List<ComPtr<ID3D11RenderTargetView>> m_SlicesRenderTargetViews;

		DXGI_FORMAT m_Format = DXGI_FORMAT_UNKNOWN;
		uint32_t m_Width = 0;
		uint32_t m_Height = 0;
		uint32_t m_Depth = 0;
		uint32_t m_ArraySize = 0;
		uint32_t m_MipLevels = 1;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		static inline uint32_t s_MaxIndex = 0;
		uint32_t m_Index;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};
}