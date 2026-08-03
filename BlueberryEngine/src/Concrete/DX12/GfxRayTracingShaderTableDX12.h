#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;

	class GfxRayTracingShaderTableDX12
	{
	public:
		GfxRayTracingShaderTableDX12() = default;
		GfxRayTracingShaderTableDX12(GfxDeviceDX12* device);
		GfxRayTracingShaderTableDX12(GfxDeviceDX12* device, size_t initialSize);

		void Clear();
		void AddRecord(void* shaderIdentifier, size_t shaderIndentifierSize, void* rootArguments, size_t rootArgumentsSize);
		void Build();

		ID3D12Resource* GetResource() const;
		size_t GetSize() const;
		size_t GetRecordCount() const;

	private:
		ComPtr<ID3D12Resource> m_Resource;
		size_t m_Size = 0;
		size_t m_RecordCount = 0;
		ByteData m_Data;

		GfxDeviceDX12* m_GfxDevice = nullptr;

		friend class GfxDeviceDX12;
	};
}