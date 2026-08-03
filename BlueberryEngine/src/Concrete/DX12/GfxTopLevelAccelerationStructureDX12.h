#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxTopLevelAccelerationStructure.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"
#include "GfxDescriptorHeapDX12.h"
#include "GfxBottomLevelAccelerationStructureDX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;
	class GfxRayTracingShader;

	struct GfxRayTracingInstanceDataDX12
	{
		GfxBottomLevelAccelerationStructure* bottomLevelAccelerationStructure;
		GfxRayTracingGeometryData geometryData;
		Material* material;
	};

	class GfxTopLevelAccelerationStructureDX12 : public GfxTopLevelAccelerationStructure
	{
	public:
		GfxTopLevelAccelerationStructureDX12(GfxDeviceDX12* device);
		virtual ~GfxTopLevelAccelerationStructureDX12() final;

		virtual void Add(GfxBottomLevelAccelerationStructure* accelerationStructure, const List<ObjectPtr<Material>>& materials, const Matrix& transform) final;
		virtual void Clear() final;

		void Build();

		ID3D12Resource* GetResource() const;
		const GfxHandleDX12& GetShaderResourceView() const;

	private:
		ComPtr<ID3D12Resource> m_Resource;
		GfxHandleDX12 m_ShaderResourceView;
		size_t m_ResourceSize = 0;

		ComPtr<ID3D12Resource> m_ScratchResource;
		size_t m_ScratchResourceSize = 0;
		List<GfxRayTracingInstanceDataDX12> m_InstanceDatas;
		List<D3D12_RAYTRACING_INSTANCE_DESC> m_InstanceDescs;

		GfxDeviceDX12* m_GfxDevice;
		ID3D12Device* m_Device;

		friend class GfxRayTracingRenderStateCacheDX12;
	};
}