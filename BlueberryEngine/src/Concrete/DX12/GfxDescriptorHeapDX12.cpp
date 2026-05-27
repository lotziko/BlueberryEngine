#include "GfxDescriptorHeapDX12.h"

#include "GfxDeviceDX12.h"
#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	bool GfxHandleDX12::IsInvalid() const
	{
		return m_Index == UINT32_MAX;
	}

	void GfxHandleDX12::Free()
	{
		if (m_Heap != nullptr && m_Index != UINT32_MAX)
		{
			m_Heap->m_FreeBlocks.push_back({ m_Index, 1 });
			m_Index = UINT32_MAX;
		}
	}

	GfxDescriptorHeapDX12::GfxDescriptorHeapDX12(GfxDeviceDX12* device) : m_GfxDevice(device), m_Device(device->GetDevice())
	{
	}

	bool GfxDescriptorHeapDX12::Initialize(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t descriptorsCount)
	{
		m_Type = type;
		m_DescriptorsCount = descriptorsCount;

		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
		heapDesc.NumDescriptors = descriptorsCount;
		heapDesc.Type = type;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		
		HRESULT hr = m_Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_Heap));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating heap."));
			return false;
		}

		m_StartCPU = m_Heap->GetCPUDescriptorHandleForHeapStart();
		m_StartGPU = m_Heap->GetGPUDescriptorHandleForHeapStart();
		m_HandleIncrement = m_Device->GetDescriptorHandleIncrementSize(type);

		Free(0, descriptorsCount);
		return true;
	}

	uint32_t GfxDescriptorHeapDX12::GetIndex(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle) const
	{
		return static_cast<uint32_t>((cpuHandle.ptr - m_StartCPU.ptr) / m_HandleIncrement);
	}

	D3D12_CPU_DESCRIPTOR_HANDLE GfxDescriptorHeapDX12::GetCPU(uint32_t index) const
	{
		D3D12_CPU_DESCRIPTOR_HANDLE handle;
		handle.ptr = m_StartCPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement);
		return handle;
	}

	D3D12_GPU_DESCRIPTOR_HANDLE GfxDescriptorHeapDX12::GetGPU(uint32_t index) const
	{
		D3D12_GPU_DESCRIPTOR_HANDLE handle;
		handle.ptr = m_StartGPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement);
		return handle;
	}

	void GfxDescriptorHeapDX12::GetCPU(D3D12_CPU_DESCRIPTOR_HANDLE* cpuHandle, uint32_t index) const
	{
		cpuHandle->ptr = m_StartCPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement);
	}

	void GfxDescriptorHeapDX12::GetGPU(D3D12_GPU_DESCRIPTOR_HANDLE* gpuHandle, uint32_t index) const
	{
		gpuHandle->ptr = m_StartGPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement);
	}

	void GfxDescriptorHeapDX12::Allocate(GfxHandleDX12* handle)
	{
		handle->m_Heap = this;
		handle->m_Index = Allocate();
	}

	uint32_t GfxDescriptorHeapDX12::Allocate(uint32_t size)
	{
		if (m_FreeBlocks.size() == 0)
		{
			Resize(m_DescriptorsCount * 2);
		}

		for (auto it = m_FreeBlocks.begin(); it != m_FreeBlocks.end(); ++it)
		{
			if (it->size >= size)
			{
				uint32_t offset = it->offset;

				it->offset += size;
				it->size -= size;

				if (it->size == 0)
				{
					it = m_FreeBlocks.erase(it);
				}

				return offset;
			}
		}
		return UINT32_MAX;
	}

	void GfxDescriptorHeapDX12::Free(uint32_t offset, uint32_t size)
	{
		m_FreeBlocks.insert(m_FreeBlocks.begin(), { offset, size });
	}

	ID3D12DescriptorHeap* GfxDescriptorHeapDX12::GetHeap() const
	{
		return m_Heap.Get();
	}

	void GfxDescriptorHeapDX12::Resize(uint32_t descriptorsCount)
	{
		if (descriptorsCount <= m_DescriptorsCount || m_Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || m_Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
		{
			return;
		}
		
		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
		heapDesc.NumDescriptors = descriptorsCount;
		heapDesc.Type = m_Type;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

		ComPtr<ID3D12DescriptorHeap> oldHeap = m_Heap;
		HRESULT hr = m_Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_Heap));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating heap."));
			return;
		}

		m_GfxDevice->WaitForGPU();
		m_GfxDevice->Reset();

		m_StartCPU = m_Heap->GetCPUDescriptorHandleForHeapStart();
		m_StartGPU = m_Heap->GetGPUDescriptorHandleForHeapStart();
		m_HandleIncrement = m_Device->GetDescriptorHandleIncrementSize(m_Type);

		m_Device->CopyDescriptorsSimple(m_DescriptorsCount, m_StartCPU, oldHeap->GetCPUDescriptorHandleForHeapStart(), m_Type);
		m_FreeBlocks.push_back({ m_DescriptorsCount, descriptorsCount - m_DescriptorsCount });
		m_DescriptorsCount = descriptorsCount;
	}

	GfxDescriptorRingHeapDX12::GfxDescriptorRingHeapDX12(GfxDeviceDX12* device) : m_Device(device->GetDevice())
	{
	}

	bool GfxDescriptorRingHeapDX12::Initialize(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t temporaryCount, uint32_t persistentCount)
	{
		m_Type = type;
		m_PersistentCount = persistentCount;
		m_TemporaryCount = temporaryCount;
		m_TemporaryOffset = persistentCount;

		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
		heapDesc.NumDescriptors = temporaryCount + persistentCount;
		heapDesc.Type = type;
		heapDesc.Flags = (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

		HRESULT hr = m_Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_Heap));
		
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating ring heap."));
			return false;
		}

		m_StartCPU = m_Heap->GetCPUDescriptorHandleForHeapStart();
		m_StartGPU = m_Heap->GetGPUDescriptorHandleForHeapStart();
		m_HandleIncrement = m_Device->GetDescriptorHandleIncrementSize(type);

		return true;
	}

	bool GfxDescriptorRingHeapDX12::CanAllocateTemporary(uint32_t size)
	{
		return m_TemporaryOffset + size <= m_TemporaryCount;
	}

	GfxRingHandleDX12 GfxDescriptorRingHeapDX12::AllocateTemporary(uint32_t size)
	{
		if (m_TemporaryOffset + size > m_TemporaryCount)
		{
			m_TemporaryOffset = m_PersistentCount;
		}
		GfxRingHandleDX12 handle;
		handle.m_CpuHandle.ptr = m_StartCPU.ptr + static_cast<SIZE_T>(m_TemporaryOffset * m_HandleIncrement);
		handle.m_GpuHandle.ptr = m_StartGPU.ptr + static_cast<SIZE_T>(m_TemporaryOffset * m_HandleIncrement);
		handle.m_Index = m_TemporaryOffset;
		m_TemporaryOffset += size;
		return handle;
	}

	uint32_t GfxDescriptorRingHeapDX12::GetOffset() const
	{
		return m_TemporaryOffset;
	}

	GfxRingHandleDX12 GfxDescriptorRingHeapDX12::AllocatePersistent(uint32_t size)
	{
		if (m_PersistentOffset + size > m_PersistentCount)
		{
			BB_ERROR("Can't allocate descriptor.");
			return {};
		}
		GfxRingHandleDX12 handle;
		handle.m_CpuHandle.ptr = m_StartCPU.ptr + static_cast<SIZE_T>(m_PersistentOffset * m_HandleIncrement);
		handle.m_GpuHandle.ptr = m_StartGPU.ptr + static_cast<SIZE_T>(m_PersistentOffset * m_HandleIncrement);
		handle.m_Index = m_PersistentOffset;
		m_PersistentOffset += size;
		return handle;
	}

	void GfxDescriptorRingHeapDX12::Reset()
	{
		m_TemporaryOffset = m_PersistentCount;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE GfxDescriptorRingHeapDX12::GetCPU(uint32_t index) const
	{
		return { m_StartCPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement) };
	}

	D3D12_GPU_DESCRIPTOR_HANDLE GfxDescriptorRingHeapDX12::GetGPU(uint32_t index) const
	{
		return { m_StartGPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement) };
	}

	ID3D12DescriptorHeap* GfxDescriptorRingHeapDX12::GetHeap() const
	{
		return m_Heap.Get();
	}
}