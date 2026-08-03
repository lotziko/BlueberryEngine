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
		const GfxHandleDX12& GetShaderResourceView() const;
		const GfxHandleDX12& GetRenderTargetView() const;
		const GfxHandleDX12& GetRenderTargetView(uint32_t arraySlice, uint32_t mipSlice);
		const GfxHandleDX12& GetDepthStencilView() const;
		const GfxHandleDX12& GetUnorderedAccessView() const;
		const GfxHandleDX12& GetRingShaderResourceView();
		uint8_t GetSampler() const;
		void SetSampler(uint8_t sampler);
		const DXGI_FORMAT GetDxgiFormat() const;

		virtual void* GetHandle() override;

		virtual void GetData(void* data, const Rectangle& area) override;
		virtual void GetData(void* data) override;
		virtual void SetData(void* data, size_t size) override;

		virtual void SetWrapMode(WrapMode wrapMode) override;
		virtual void SetFilterMode(FilterMode filterMode) override;
		virtual void SetName(const String& name) override;

		D3D12_RESOURCE_STATES GetState() const;
		void SetState(D3D12_RESOURCE_STATES state);
		void SetUAVState();

		static GfxTextureDX12* Get(uint32_t index);

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

		GfxHandleDX12 m_RingShaderResourceView;

		List<GfxHandleDX12> m_SlicesRenderTargetViews;

		GfxDeviceDX12* m_GfxDevice;
		ID3D12Device* m_Device;

		DXGI_FORMAT m_DxgiFormat = DXGI_FORMAT_UNKNOWN;
		D3D12_RESOURCE_STATES m_State = D3D12_RESOURCE_STATE_COMMON;
		uint64_t m_UnorderedAccessGeneration = 0;

		static GfxPointerCache<GfxTextureDX12> s_PointerCache;
	};
}