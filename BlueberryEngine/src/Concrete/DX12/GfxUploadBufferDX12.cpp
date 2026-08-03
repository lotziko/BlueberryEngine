#include "GfxUploadBufferDX12.h"

#include "GfxDeviceDX12.h"
#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	#define UPLOAD_PAGE_SIZE 4 * 1024 * 1024

	GfxUploadBufferPageDX12::GfxUploadBufferPageDX12(GfxDeviceDX12* device, uint64_t size)
	{
		m_GfxDevice = device;
		m_Size = size;
		HRESULT hr = device->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(size), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_Resource));
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create resource."));
			return;
		}
		m_Resource->Map(0, nullptr, &m_Ptr);
	}

	void GfxUploadBufferPageDX12::Free()
	{
		if (m_Ptr != nullptr)
		{
			m_Resource->Unmap(0, nullptr);
			m_Ptr = nullptr;
		}
		m_GfxDevice->Release(m_Resource);
	}

	GfxUploadBufferDX12::GfxUploadBufferDX12(GfxDeviceDX12* device) : m_GfxDevice(device), m_Device(device->GetDevice()), m_CommandList(device->GetCommandList())
	{
	}

	void GfxUploadBufferDX12::UploadBuffer(ID3D12Resource* resource, const void* data, uint64_t size, uint64_t alignment)
	{
		GfxUploadBufferAllocationDX12 allocation = Allocate(size, alignment);
		memcpy(allocation.ptr, data, size);
		m_CommandList->CopyBufferRegion(resource, 0, allocation.page->m_Resource.Get(), allocation.offset, size);
	}

	void GfxUploadBufferDX12::UploadBuffer(D3D12_GPU_VIRTUAL_ADDRESS& adress, const void* data, uint64_t size, uint64_t alignment)
	{
		GfxUploadBufferAllocationDX12 allocation = Allocate(size, alignment);
		memcpy(allocation.ptr, data, size);
		adress = allocation.page->m_Resource->GetGPUVirtualAddress() + allocation.offset;
	}

	void GfxUploadBufferDX12::UploadTexture(ID3D12Resource* resource, D3D12_SUBRESOURCE_DATA* subresourceData, UINT subresourceCount, uint64_t alignment)
	{
		List<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(subresourceCount);
		List<UINT> numRows(subresourceCount);
		List<UINT64> rowSizes(subresourceCount);

		UINT64 size;
		m_Device->GetCopyableFootprints(&resource->GetDesc(), 0, subresourceCount, 0, layouts.data(), numRows.data(), rowSizes.data(), &size);
		GfxUploadBufferAllocationDX12 allocation = Allocate(size, alignment);
		
		uint8_t* dstBase = allocation.ptr;
		for (UINT i = 0; i < subresourceCount; ++i)
		{
			const D3D12_SUBRESOURCE_DATA& src = subresourceData[i];
			const uint8_t* srcPtr = static_cast<const uint8_t*>(src.pData);
			
			UINT numRow = numRows[i];
			UINT64 rowSize = rowSizes[i];

			auto& layout = layouts[i];
			uint8_t* dstPtr = dstBase + layout.Offset;
			UINT dstRowPitch = layout.Footprint.RowPitch;
			
			for (UINT z = 0; z < layout.Footprint.Depth; ++z)
			{
				const uint8_t* sliceSrcPtr = srcPtr + src.SlicePitch * z;
				uint8_t* sliceDstPtr = dstPtr + (dstRowPitch * numRow) * z;
				for (UINT y = 0; y < numRow; ++y)
				{
					memcpy(sliceDstPtr + y * dstRowPitch, sliceSrcPtr + y * src.RowPitch, rowSize);
				}
			}

			D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
			dstLocation.pResource = resource;
			dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			dstLocation.SubresourceIndex = i;

			D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
			srcLocation.pResource = allocation.page->m_Resource.Get();
			srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
			srcLocation.PlacedFootprint = layout;
			srcLocation.PlacedFootprint.Offset += allocation.offset;

			m_CommandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);
		}
	}

	void GfxUploadBufferDX12::UpdateGeneration(uint64_t generation)
	{
		m_Generation = generation;
		if (m_UsedPages.size() > 0)
		{
			for (auto it = m_UsedPages.begin(); it != m_UsedPages.end();)
			{
				GfxUploadBufferPageDX12* page = *it;
				if (page->m_Generation + 2 < m_Generation)
				{
					if (page == m_CurrentPage)
					{
						m_CurrentPage = nullptr;
					}
					it = m_UsedPages.erase(it);
					m_FreePages.push_back(page);
				}
				else
				{
					++it;
				}
			}
		}

		if (m_FreePages.size() > 0)
		{
			for (auto it = m_FreePages.begin(); it != m_FreePages.end();)
			{
				GfxUploadBufferPageDX12* page = *it;
				if (page->m_Generation + 60 < m_Generation)
				{
					if (page == m_CurrentPage)
					{
						m_CurrentPage = nullptr;
					}
					it = m_FreePages.erase(it);
					page->Free();
					delete page;
				}
				else
				{
					++it;
				}
			}
		}
	}

	GfxUploadBufferAllocationDX12 GfxUploadBufferDX12::Allocate(uint64_t size, uint64_t alignment)
	{
		uint64_t pageSize = (size <= UPLOAD_PAGE_SIZE) ? UPLOAD_PAGE_SIZE : size;
		uint64_t alignedSize = Math::NextDivisableBy(size, alignment);

		m_CurrentOffset = Math::NextDivisableBy(m_CurrentOffset, alignment);

		if (m_CurrentPage != nullptr && m_CurrentOffset + alignedSize > m_CurrentPage->m_Size)
		{
			m_CurrentPage = nullptr;
		}

		if (m_CurrentPage == nullptr)
		{
			m_CurrentPage = AllocatePage(pageSize);
			m_CurrentOffset = 0;
		}

		m_CurrentPage->m_Generation = m_Generation;

		GfxUploadBufferAllocationDX12 result = {};
		result.page = m_CurrentPage;
		result.ptr = static_cast<uint8_t*>(m_CurrentPage->m_Ptr) + m_CurrentOffset;
		result.offset = m_CurrentOffset;
		result.size = size;
		result.generation = m_Generation;

		m_CurrentOffset += size;
		return result;
	}

	GfxUploadBufferPageDX12* GfxUploadBufferDX12::AllocatePage(uint64_t size)
	{
		size_t freePageIndex = UINT64_MAX;
		for (size_t i = 0; i < m_FreePages.size(); i++)
		{
			if (m_FreePages[i]->m_Size == size)
			{
				freePageIndex = i;
				break;
			}
		}
		GfxUploadBufferPageDX12* page;
		if (freePageIndex == UINT64_MAX)
		{
			page = new GfxUploadBufferPageDX12(m_GfxDevice, size);
		}
		else
		{
			page = m_FreePages[freePageIndex];
			m_FreePages.erase(m_FreePages.begin() + freePageIndex);
		}
		m_UsedPages.push_back(page);
		return page;
	}
}
