#pragma once
#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxTexture.h"

namespace Blueberry
{
	class GfxTextureDX11 : public GfxTexture
	{
	public:
		GfxTextureDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxTextureDX11() final;
		
		bool Create(const TextureProperties& properties);

		ID3D11ShaderResourceView* GetSRV() const;
		ID3D11RenderTargetView* GetRTV() const;

		virtual uint32_t GetWidth() const override;
		virtual uint32_t GetHeight() const override;
		virtual void* GetHandle() override;

		virtual void SetData(void* data) override;

	private:
		uint32_t GetQualityLevel(const DXGI_FORMAT& format, const uint32_t& antiAliasing);
		bool InitializeResource(D3D11_SUBRESOURCE_DATA* subresourceData, const TextureProperties& properties);
		bool InitializeRenderTarget(const TextureProperties& properties);
		bool InitializeDepthStencil(const TextureProperties& properties);

	private:
		ComPtr<ID3D11Texture2D> m_Texture;
		ComPtr<ID3D11ShaderResourceView> m_ResourceView;
		ComPtr<ID3D11SamplerState> m_SamplerState;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;
		ComPtr<ID3D11DepthStencilView> m_DepthStencilView;
		ComPtr<ID3D11Texture2D> m_StagingTexture;

		uint32_t m_Width;
		uint32_t m_Height;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
	};
}