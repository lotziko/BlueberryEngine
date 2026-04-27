#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"
#include "GfxPointerCacheDX11.h"

namespace Blueberry
{
	class GfxBufferDX11 final : public GfxBuffer
	{
	public:
		GfxBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxBufferDX11() final;

		bool Initialize(const BufferProperties& properties);

		virtual void* Map() final;
		virtual void Unmap() final;

		virtual void GetData(void* data) final;
		virtual void SetData(const void* data, size_t size) final;

		virtual uint32_t GetElementSize() const final;
		virtual uint32_t GetElementCount() const final;

	private:
		bool Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, const BufferProperties& properties);

	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;
		ComPtr<ID3D11ShaderResourceView> m_ShaderResourceView = nullptr;
		ComPtr<ID3D11UnorderedAccessView> m_UnorderedAccessView = nullptr;
		ComPtr<ID3D11Buffer> m_StagingBuffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		uint32_t m_ElementSize = 0;
		uint32_t m_ElementCount = 0;
		bool m_IsConstant = false;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;

		static GfxPointerCacheDX11<GfxBufferDX11> s_PointerCache;
	};
}