#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;
	class GfxDescriptorHeapDX12;

	class GfxHandleDX12
	{
	public:
		GfxHandleDX12() = default;

		D3D12_CPU_DESCRIPTOR_HANDLE GetCPU() const;
		D3D12_GPU_DESCRIPTOR_HANDLE GetGPU() const;

		inline uint32_t GetIndex() const
		{
			return m_Index;
		}

		bool IsInvalid() const;
		void Free();

	private:
		GfxDescriptorHeapDX12* m_Heap = nullptr;
		uint32_t m_Index = UINT32_MAX;
		uint32_t m_Size = 0;
		bool m_IsPersistent = false;

		friend class GfxDescriptorHeapDX12;
	};

	class GfxDescriptorHeapDX12
	{
	public:
		GfxDescriptorHeapDX12() = default;
		GfxDescriptorHeapDX12(GfxDeviceDX12* device);

		bool Initialize(D3D12_DESCRIPTOR_HEAP_TYPE type, bool isShaderVisible, uint32_t persistentDescriptorsCount, uint32_t temporaryDescriptorsCount);

		inline uint32_t GetIndex(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle) const
		{
			return static_cast<uint32_t>((cpuHandle.ptr - m_StartCPU.ptr) / m_HandleIncrement);
		}

		inline D3D12_CPU_DESCRIPTOR_HANDLE GetCPU(uint32_t index) const
		{
			D3D12_CPU_DESCRIPTOR_HANDLE handle = {};
			handle.ptr = m_StartCPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement);
			return handle;
		}

		inline D3D12_GPU_DESCRIPTOR_HANDLE GetGPU(uint32_t index) const
		{
			D3D12_GPU_DESCRIPTOR_HANDLE handle = {};
			handle.ptr = m_StartGPU.ptr + static_cast<SIZE_T>(index * m_HandleIncrement);
			return handle;
		}

		inline D3D12_CPU_DESCRIPTOR_HANDLE GetStartCPU() const
		{
			return m_StartCPU;
		}

		inline D3D12_GPU_DESCRIPTOR_HANDLE GetStartGPU() const
		{
			return m_StartGPU;
		}
		
		GfxHandleDX12 AllocatePersistent(uint32_t size = 1);
		GfxHandleDX12 AllocateTemporary(uint32_t size = 1);
		void Free(GfxHandleDX12 handle);
		void Free(D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle);
		
		ID3D12DescriptorHeap* GetHeap() const;

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
		uint32_t m_PersistentCount = 0;
		List<HeapBlock> m_PersistentFreeBlocks;
		uint32_t m_TemporaryCount = 0;
		uint32_t m_TemporaryOffset = 0;

		GfxDeviceDX12* m_GfxDevice;
		ID3D12Device* m_Device;

		friend class GfxHandleDX12;
	};
}