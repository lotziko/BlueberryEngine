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

		ID3D11Resource* GetResource() const;
		ID3D11ShaderResourceView* GetShaderResourceView() const;
		ID3D11RenderTargetView* GetRenderTargetView() const;
		ID3D11RenderTargetView* GetRenderTargetView(uint32_t arraySlice, uint32_t mipSlice);
		ID3D11DepthStencilView* GetDepthStencilView() const;
		ID3D11UnorderedAccessView* GetUnorderedAccessView() const;
		ID3D11SamplerState* GetSamplerState() const;
		void SetSamplerState(ID3D11SamplerState* samplerState);
		const DXGI_FORMAT GetDxgiFormat() const;

		virtual void* GetHandle() override;

		virtual void GetData(void* data, const Rectangle& area) override;
		virtual void GetData(void* data) override;
		virtual void SetData(void* data, size_t size) override;

		virtual void SetWrapMode(WrapMode wrapMode) override;
		virtual void SetFilterMode(FilterMode filterMode) override;
		virtual void SetName(const String& name) override;

		static GfxTextureDX11* Get(uint32_t index);

	private:
		bool Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, uint32_t subresourceCount, const TextureProperties& properties);

	private:
		ComPtr<ID3D11Resource> m_Resource;
		ComPtr<ID3D11ShaderResourceView> m_ShaderResourceView;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;
		ComPtr<ID3D11DepthStencilView> m_DepthStencilView;
		ComPtr<ID3D11UnorderedAccessView> m_UnorderedAccessView;
		ComPtr<ID3D11Resource> m_StagingTexture;
		ComPtr<ID3D11SamplerState> m_SamplerState;

		List<ComPtr<ID3D11RenderTargetView>> m_SlicesRenderTargetViews;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		DXGI_FORMAT m_DxgiFormat = DXGI_FORMAT_UNKNOWN;

		static GfxPointerCache<GfxTextureDX11> s_PointerCache;
	};
}