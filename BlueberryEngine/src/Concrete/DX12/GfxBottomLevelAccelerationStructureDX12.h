#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxBottomLevelAccelerationStructure.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;

	struct GfxRayTracingGeometryData
	{
		D3D12_GPU_VIRTUAL_ADDRESS vertexBufferAddress;
		D3D12_GPU_VIRTUAL_ADDRESS indexBufferAddress;
		UINT vertexStride;
		UINT normalOffset;
		UINT uv0Offset;
	};

	class GfxBottomLevelAccelerationStructureDX12 : public GfxBottomLevelAccelerationStructure
	{
	public:
		GfxBottomLevelAccelerationStructureDX12(GfxDeviceDX12* device);
		virtual ~GfxBottomLevelAccelerationStructureDX12() final;

		bool Initialize(const BottomLevelAccelerationStructureProperties& properties);

	private:
		ComPtr<ID3D12Resource> m_Resource;
		ComPtr<ID3D12Resource> m_ScratchResource;

		GfxDeviceDX12* m_GfxDevice;

		GfxRayTracingGeometryData m_GeometryData = {};

		friend class GfxTopLevelAccelerationStructureDX12;
	};
}