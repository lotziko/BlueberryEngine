#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"
#include "GfxDescriptorHeapDX12.h"
#include "..\..\Blueberry\Graphics\GfxPointerCache.h"

namespace Blueberry
{
	class GfxDeviceDX12;

	class GfxBufferDX12 final : public GfxBuffer
	{
	public:
		GfxBufferDX12(GfxDeviceDX12* device);
		virtual ~GfxBufferDX12() final;

		bool Initialize(const BufferProperties& properties);

		virtual void GetData(void* data) final;
		virtual void SetData(const void* data, size_t size) final;

		D3D12_VERTEX_BUFFER_VIEW GetVertexView();
		D3D12_INDEX_BUFFER_VIEW GetIndexView();

		virtual uint32_t GetElementSize() const final;
		virtual uint32_t GetElementCount() const final;

	private:
		bool Initialize(D3D12_SUBRESOURCE_DATA* subresourceData, const BufferProperties& properties);

	private:
		ComPtr<ID3D12Resource> m_Resource;
		GfxHandleDX12 m_ShaderResourceView;
		GfxHandleDX12 m_UnorderedAccessView;
		GfxHandleDX12 m_ConstantBufferView;

		ID3D12Device* m_Device;
		GfxDeviceDX12* m_GfxDevice;

		uint32_t m_ElementSize = 0;
		uint32_t m_ElementCount = 0;
		bool m_IsConstant = false;
		D3D12_RESOURCE_STATES m_State = D3D12_RESOURCE_STATE_COMMON;

		static GfxPointerCache<GfxBufferDX12> s_PointerCache;

		friend class GfxDeviceDX12;
		friend class GfxRenderStateCacheDX12;
		friend class GfxComputeRenderStateCacheDX12;
	};
}