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

		ID3D12Resource* GetResource();
		const GfxHandleDX12& GetShaderResourceView() const;
		const GfxHandleDX12& GetUnorderedAccessView() const;
		const GfxHandleDX12& GetConstantBufferView() const;
		D3D12_VERTEX_BUFFER_VIEW GetVertexView();
		D3D12_INDEX_BUFFER_VIEW GetIndexView();

		D3D12_RESOURCE_STATES GetState() const;
		void SetState(D3D12_RESOURCE_STATES state);
		void SetUAVState();

		static GfxBufferDX12* Get(uint32_t index);

	private:
		bool Initialize(D3D12_SUBRESOURCE_DATA* subresourceData, const BufferProperties& properties);

	private:
		ComPtr<ID3D12Resource> m_Resource;
		GfxHandleDX12 m_ShaderResourceView;
		GfxHandleDX12 m_UnorderedAccessView;
		GfxHandleDX12 m_ConstantBufferView;

		GfxDeviceDX12* m_GfxDevice;
		ID3D12Device* m_Device;

		D3D12_RESOURCE_STATES m_State = D3D12_RESOURCE_STATE_COMMON;
		uint64_t m_Generation = 0;

		static GfxPointerCache<GfxBufferDX12> s_PointerCache;
	};
}