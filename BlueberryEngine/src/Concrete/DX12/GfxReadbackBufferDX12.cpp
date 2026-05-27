#include "GfxReadbackBufferDX12.h"

#include "GfxDeviceDX12.h"
#include "..\Windows\WindowsHelper.h"
#include "..\Windows\DxgiHelper.h"

namespace Blueberry
{
	#define READBACK_PAGE_SIZE 4 * 1024 * 1024ull

	GfxReadbackBufferDX12::GfxReadbackBufferDX12(GfxDeviceDX12* device) : m_GfxDevice(device), m_Device(device->GetDevice()), m_CommandList(device->GetCommandList())
	{
	}

	void GfxReadbackBufferDX12::ReadBuffer(ID3D12Resource* resource, void* data, uint64_t size)
	{
		ResizeIfNeeded(size);
		m_CommandList->CopyBufferRegion(m_Resource.Get(), 0, resource, 0, size);
		m_GfxDevice->WaitForGPU();
		m_GfxDevice->Reset();
		memcpy(data, m_Ptr, size);
	}

	void GfxReadbackBufferDX12::ReadTexture(ID3D12Resource* resource, void* data, const Rectangle& area)
	{
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
		UINT numRows;
		UINT64 rowSizes;
		UINT64 size;
		m_Device->GetCopyableFootprints(&resource->GetDesc(), 0, 1, 0, &layout, &numRows, &rowSizes, &size);
		ResizeIfNeeded(size);

		D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
		dstLocation.pResource = m_Resource.Get();
		dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
		dstLocation.PlacedFootprint = layout;

		D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
		srcLocation.pResource = resource;
		srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		srcLocation.SubresourceIndex = 0;

		m_CommandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);
		m_GfxDevice->WaitForGPU();
		m_GfxDevice->Reset();

		for (int i = 0; i < area.height; i++)
		{
			size_t pixelSize = layout.Footprint.RowPitch / layout.Footprint.Width;
			size_t offset = ((area.y + i) * layout.Footprint.Width + area.x) * pixelSize;
			char* copyPtr = static_cast<char*>(m_Ptr) + offset;
			char* targetPtr = static_cast<char*>(data) + (area.width * pixelSize * i);
			memcpy(targetPtr, copyPtr, area.width * pixelSize);
		}
	}

	void GfxReadbackBufferDX12::ReadTexture(ID3D12Resource* resource, void* data, uint32_t subresourceCount)
	{
		List<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(subresourceCount);
		List<UINT> numRows(subresourceCount);
		List<UINT64> rowSizes(subresourceCount);
		UINT64 size;
		D3D12_RESOURCE_DESC* desc = &resource->GetDesc();
		m_Device->GetCopyableFootprints(desc, 0, 1, 0, layouts.data(), numRows.data(), rowSizes.data(), &size);
		ResizeIfNeeded(size);

		uint32_t bytesPerPixel = DxgiHelper::GetBitsPerPixel(desc->Format) / 8;
		for (uint32_t i = 0; i < subresourceCount; ++i)
		{
			D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
			dstLocation.pResource = m_Resource.Get();
			dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
			dstLocation.PlacedFootprint = layouts[i];

			D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
			srcLocation.pResource = resource;
			srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			srcLocation.SubresourceIndex = i;

			m_CommandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);
		}

		m_GfxDevice->WaitForGPU();
		m_GfxDevice->Reset();

		uint8_t* ptr = static_cast<uint8_t*>(data);
		for (uint32_t i = 0; i < subresourceCount; ++i)
		{
			auto& layout = layouts[i];
			uint8_t* src = static_cast<uint8_t*>(m_Ptr) + layout.Offset;
			uint32_t dataSize = layout.Footprint.Width * bytesPerPixel;
			for (uint32_t j = 0; j < numRows[i]; ++j)
			{
				memcpy(ptr, src, dataSize);
				src += layout.Footprint.RowPitch;
				ptr += dataSize;
			}
		}
	}

	void GfxReadbackBufferDX12::ResizeIfNeeded(uint64_t size)
	{
		if (size > m_Size)
		{
			if (m_Resource != nullptr)
			{
				m_Resource->Unmap(0, nullptr);
				m_GfxDevice->Release(m_Resource);
				m_Resource = nullptr;
			}
			m_Size = std::max(size, READBACK_PAGE_SIZE);
			HRESULT hr = m_Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(size), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_Resource));
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create resource."));
				return;
			}
			m_Resource->Map(0, nullptr, &m_Ptr);
		}
	}
}