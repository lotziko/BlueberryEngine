#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"
#include "..\..\Blueberry\Graphics\GfxPointerCache.h"

namespace Blueberry
{
	class GfxBufferDX11 final : public GfxBuffer
	{
	public:
		GfxBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxBufferDX11() final;

		bool Initialize(const BufferProperties& properties);

		ID3D11Buffer* GetBuffer() const;
		ID3D11ShaderResourceView* GetShaderResourceView() const;
		ID3D11UnorderedAccessView* GetUnorderedAccessView() const;

		virtual void GetData(void* data) final;
		virtual void SetData(const void* data, size_t size) final;

		static GfxBufferDX11* Get(uint32_t index);

	private:
		bool Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, const BufferProperties& properties);

	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;
		ComPtr<ID3D11ShaderResourceView> m_ShaderResourceView = nullptr;
		ComPtr<ID3D11UnorderedAccessView> m_UnorderedAccessView = nullptr;
		ComPtr<ID3D11Buffer> m_StagingBuffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		static GfxPointerCache<GfxBufferDX11> s_PointerCache;
	};
}