#pragma once
#include "Blueberry\Graphics\GfxTexture.h"

namespace Blueberry
{
	class GfxTextureDX11 : public GfxTexture
	{
	public:
		GfxTextureDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxTextureDX11() final = default;
		
		bool Create(const UINT& width, const UINT& height, bool isRenderTarget);
		bool Load(const std::string& path);

		virtual UINT GetWidth() const override;
		virtual UINT GetHeight() const override;
		virtual void* GetHandle() override;

		void Clear(const Color& color);

	private:
		bool Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, bool isRenderTarget);

	private:
		ComRef<ID3D11Texture2D> m_Texture;
		ComRef<ID3D11ShaderResourceView> m_ResourceView;
		ComRef<ID3D11SamplerState> m_SamplerState;
		ComRef<ID3D11RenderTargetView> m_RenderTargetView;

		UINT m_Width;
		UINT m_Height;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
	};
}