#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;

	class GfxUploadBufferPageDX12
	{
	public:
		GfxUploadBufferPageDX12(GfxDeviceDX12* device, uint64_t size);
		void Free();

	private:
		ComPtr<ID3D12Resource> m_Resource;
		GfxDeviceDX12* m_GfxDevice = nullptr;
		uint64_t m_Size = 0;
		uint64_t m_Generation = 0;
		void* m_Ptr = nullptr;

		friend class GfxUploadBufferDX12;
	};

	struct GfxUploadBufferAllocationDX12
	{
		GfxUploadBufferPageDX12* page = nullptr;
		uint8_t* ptr = nullptr;
		uint64_t offset = 0;
		uint64_t size = 0;
		uint64_t generation = 0;
	};

	class GfxUploadBufferDX12
	{
	public:
		GfxUploadBufferDX12() = default;
		GfxUploadBufferDX12(GfxDeviceDX12* device);

		void UploadBuffer(ID3D12Resource* resource, const void* data, uint64_t size, uint64_t alignment);
		void UploadBuffer(D3D12_GPU_VIRTUAL_ADDRESS& adress, const void* data, uint64_t size, uint64_t alignment);
		void UploadTexture(ID3D12Resource* resource, D3D12_SUBRESOURCE_DATA* subresourceData, UINT subresourceCount, uint64_t alignment);
		void UpdateGeneration(uint64_t generation);

	private:
		GfxUploadBufferAllocationDX12 Allocate(uint64_t size, uint64_t alignment);
		GfxUploadBufferPageDX12* AllocatePage(uint64_t size);

	private:
		GfxDeviceDX12* m_GfxDevice;
		ID3D12Device* m_Device;
		ID3D12GraphicsCommandList* m_CommandList;

		List<GfxUploadBufferPageDX12*> m_FreePages;
		List<GfxUploadBufferPageDX12*> m_UsedPages;
		GfxUploadBufferPageDX12* m_CurrentPage = nullptr;
		uint64_t m_CurrentOffset = 0;
		uint64_t m_Generation = 0;
	};
}