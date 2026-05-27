#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;
	class GfxHandleDX12;

	class GfxDescriptorHeapDX12
	{
	public:
		GfxDescriptorHeapDX12() = default;
		GfxDescriptorHeapDX12(GfxDeviceDX12* device);

		bool Initialize(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t descriptorsCount);

		uint32_t GetIndex(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle) const;

		D3D12_CPU_DESCRIPTOR_HANDLE GetCPU(uint32_t index) const;
		D3D12_GPU_DESCRIPTOR_HANDLE GetGPU(uint32_t index) const;
		void GetCPU(D3D12_CPU_DESCRIPTOR_HANDLE* cpuHandle, uint32_t index) const;
		void GetGPU(D3D12_GPU_DESCRIPTOR_HANDLE* gpuHandle, uint32_t index) const;

		void Allocate(GfxHandleDX12* handle);
		uint32_t Allocate(uint32_t size = 1);
		void Free(uint32_t offset, uint32_t size = 1);
		
		ID3D12DescriptorHeap* GetHeap() const;

	private:
		void Resize(uint32_t descriptorsCount);

	private:
		struct HeapBlock
		{
			uint32_t offset;
			uint32_t size;
		};

		ComPtr<ID3D12DescriptorHeap> m_Heap;
		D3D12_DESCRIPTOR_HEAP_TYPE m_Type = D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES;
		uint32_t m_DescriptorsCount = 0;
		D3D12_CPU_DESCRIPTOR_HANDLE m_StartCPU = {};
		D3D12_GPU_DESCRIPTOR_HANDLE m_StartGPU = {};
		uint32_t m_HandleIncrement = 0;
		List<HeapBlock> m_FreeBlocks;

		GfxDeviceDX12* m_GfxDevice;
		ID3D12Device* m_Device;

		friend class GfxHandleDX12;
	};

	class GfxRingHandleDX12
	{
	public:
		GfxRingHandleDX12() = default;

		inline D3D12_CPU_DESCRIPTOR_HANDLE GetCPU() const
		{
			return m_CpuHandle;
		}

		inline D3D12_GPU_DESCRIPTOR_HANDLE GetGPU() const
		{
			return m_GpuHandle;
		}

		inline uint32_t GetIndex() const
		{
			return m_Index;
		}

	private:
		D3D12_CPU_DESCRIPTOR_HANDLE m_CpuHandle;
		D3D12_GPU_DESCRIPTOR_HANDLE m_GpuHandle;
		uint32_t m_Index = 0;

		friend class GfxDescriptorRingHeapDX12;
	};

	class GfxDescriptorRingHeapDX12
	{
	public:
		GfxDescriptorRingHeapDX12() = default;
		GfxDescriptorRingHeapDX12(GfxDeviceDX12* device);

		bool Initialize(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t descriptorsCount, uint32_t persistentCount);

		bool CanAllocateTemporary(uint32_t size);
		GfxRingHandleDX12 AllocateTemporary(uint32_t size = 1);
		GfxRingHandleDX12 AllocatePersistent(uint32_t size = 1);
		void Reset();

		D3D12_CPU_DESCRIPTOR_HANDLE GetCPU(uint32_t index) const;
		D3D12_GPU_DESCRIPTOR_HANDLE GetGPU(uint32_t index) const;

		ID3D12DescriptorHeap* GetHeap() const;
		uint32_t GetOffset() const;

	private:
		ComPtr<ID3D12DescriptorHeap> m_Heap;
		D3D12_DESCRIPTOR_HEAP_TYPE m_Type = D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES;
		D3D12_CPU_DESCRIPTOR_HANDLE m_StartCPU = {};
		D3D12_GPU_DESCRIPTOR_HANDLE m_StartGPU = {};
		uint32_t m_HandleIncrement = 0;
		uint32_t m_PersistentCount = 0;
		uint32_t m_PersistentOffset = 0;
		uint32_t m_TemporaryCount = 0;
		uint32_t m_TemporaryOffset = 0;

		ID3D12Device* m_Device;
	};

	class GfxHandleDX12
	{
	public:
		GfxHandleDX12() = default;

		inline D3D12_CPU_DESCRIPTOR_HANDLE GetCPU() const
		{
			return m_Heap->GetCPU(m_Index);
		}

		inline D3D12_GPU_DESCRIPTOR_HANDLE GetGPU() const
		{
			return m_Heap->GetGPU(m_Index);
		}

		bool IsInvalid() const;
		void Free();

	private:
		GfxDescriptorHeapDX12* m_Heap = nullptr;
		uint32_t m_Index = UINT32_MAX;

		friend class GfxDescriptorHeapDX12;
	};
}