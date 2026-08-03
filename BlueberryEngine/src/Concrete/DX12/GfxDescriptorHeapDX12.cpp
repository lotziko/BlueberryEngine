#include "GfxDescriptorHeapDX12.h"

#include "GfxDeviceDX12.h"
#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	D3D12_CPU_DESCRIPTOR_HANDLE GfxHandleDX12::GetCPU() const
	{
		return m_Heap->GetCPU(m_Index);
	}

	D3D12_GPU_DESCRIPTOR_HANDLE GfxHandleDX12::GetGPU() const
	{
		return m_Heap->GetGPU(m_Index);
	}

	bool GfxHandleDX12::IsInvalid() const
	{
		return m_Index == UINT32_MAX;
	}

	void GfxHandleDX12::Free()
	{
		if (m_Heap != nullptr && m_Index != UINT32_MAX)
		{
			m_Heap->Free(*this);
			m_Index = UINT32_MAX;
		}
	}

	GfxDescriptorHeapDX12::GfxDescriptorHeapDX12(GfxDeviceDX12* device) : m_GfxDevice(device), m_Device(device->GetDevice())
	{
	}

	bool GfxDescriptorHeapDX12::Initialize(D3D12_DESCRIPTOR_HEAP_TYPE type, bool isShaderVisible, uint32_t persistentDescriptorsCount, uint32_t temporaryDescriptorsCount)
	{
		m_Type = type;

		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
		heapDesc.NumDescriptors = persistentDescriptorsCount + temporaryDescriptorsCount;
		heapDesc.Type = type;
		heapDesc.Flags = isShaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		
		HRESULT hr = m_Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_Heap));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating heap."));
			return false;
		}

		m_StartCPU = m_Heap->GetCPUDescriptorHandleForHeapStart();
		if (isShaderVisible)
		{
			m_StartGPU = m_Heap->GetGPUDescriptorHandleForHeapStart();
		}
		m_HandleIncrement = m_Device->GetDescriptorHandleIncrementSize(type);
		m_PersistentCount = persistentDescriptorsCount;
		m_PersistentFreeBlocks.push_back({ 0, persistentDescriptorsCount });
		m_TemporaryCount = temporaryDescriptorsCount;
		m_TemporaryOffset = persistentDescriptorsCount;

		return true;
	}

	GfxHandleDX12 GfxDescriptorHeapDX12::AllocatePersistent(uint32_t size)
	{
		GfxHandleDX12 handle = {};
		handle.m_Heap = this;
		handle.m_Size = size;
		handle.m_IsPersistent = true;

		for (auto it = m_PersistentFreeBlocks.begin(); it != m_PersistentFreeBlocks.end(); ++it)
		{
			if (it->size >= size)
			{
				uint32_t offset = it->offset;

				it->offset += size;
				it->size -= size;

				if (it->size == 0)
				{
					it = m_PersistentFreeBlocks.erase(it);
				}

				handle.m_Index = offset;
				return handle;
			}
		}
		return {};
	}

	GfxHandleDX12 GfxDescriptorHeapDX12::AllocateTemporary(uint32_t size)
	{
		GfxHandleDX12 handle = {};
		if (m_TemporaryOffset + size > m_PersistentCount + m_TemporaryCount)
		{
			m_TemporaryOffset = m_PersistentCount;
		}
		handle.m_Heap = this;
		handle.m_Size = size;
		handle.m_Index = m_TemporaryOffset;
		m_TemporaryOffset += size;
		return handle;
	}

	void GfxDescriptorHeapDX12::Free(GfxHandleDX12 handle)
	{
		if (handle.m_IsPersistent)
		{
			m_PersistentFreeBlocks.insert(m_PersistentFreeBlocks.begin(), { handle.m_Index, handle.m_Size });
		}
	}

	void GfxDescriptorHeapDX12::Free(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle)
	{
		if (cpuHandle.ptr >= m_StartCPU.ptr)
		{
			uint32_t index = GetIndex(cpuHandle);
			if (index < m_PersistentCount)
			{
				m_PersistentFreeBlocks.insert(m_PersistentFreeBlocks.begin(), { index, 1 });
			}
		}
	}

	ID3D12DescriptorHeap* GfxDescriptorHeapDX12::GetHeap() const
	{
		return m_Heap.Get();
	}
}