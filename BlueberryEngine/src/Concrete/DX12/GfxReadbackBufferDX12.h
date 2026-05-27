#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;

	class GfxReadbackBufferDX12
	{
	public:
		GfxReadbackBufferDX12() = default;
		GfxReadbackBufferDX12(GfxDeviceDX12* device);

		void ReadBuffer(ID3D12Resource* resource, void* data, uint64_t size);
		void ReadTexture(ID3D12Resource* resource, void* data, const Rectangle& area);
		void ReadTexture(ID3D12Resource* resource, void* data, uint32_t subresourceCount);

	private:
		void ResizeIfNeeded(uint64_t size);

	private:
		GfxDeviceDX12* m_GfxDevice;
		ID3D12Device* m_Device;
		ID3D12GraphicsCommandList* m_CommandList;

		ComPtr<ID3D12Resource> m_Resource;
		size_t m_Size = 0;
		void* m_Ptr = nullptr;
	};
}