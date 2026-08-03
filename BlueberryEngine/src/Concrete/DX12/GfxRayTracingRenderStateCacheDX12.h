#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\GfxRenderStateCache.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"
#include "GfxRayTracingShaderTableDX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;
	class GfxRayTracingShader;
	class RayTracingShader;
	class GfxTopLevelAccelerationStructure;

	struct GfxRayTracingRenderStateDX12
	{
		ID3D12StateObject* stateObject;
		ID3D12StateObjectProperties* stateObjectProperties;

		D3D12_CPU_DESCRIPTOR_HANDLE constantBuffers[4];
		D3D12_CPU_DESCRIPTOR_HANDLE shaderResourceViews[4];
		D3D12_CPU_DESCRIPTOR_HANDLE unorderedAccessViews[4];
		uint8_t samplers[4];

		UINT constantBuffersCount;
		UINT shaderResourceViewsCount;
		UINT unorderedAccessViewsCount;
		UINT samplersCount;

		D3D12_GPU_VIRTUAL_ADDRESS rayGenerationShaderTableAddress;
		D3D12_GPU_VIRTUAL_ADDRESS hitGroupShaderTableAddress;
		D3D12_GPU_VIRTUAL_ADDRESS missShaderTableAddress;

		UINT64 rayGenerationShaderTableSize;
		UINT64 hitGroupShaderTableSize;
		UINT64 hitGroupShaderTableStride;
		UINT64 missShaderTableSize;
	};

	struct GfxRayTracingPipelineStateDX12
	{
		ComPtr<ID3D12StateObject> stateObject;
		ComPtr<ID3D12StateObjectProperties> stateObjectProperties;

		GfxRayTracingShaderTableDX12 rayGenerationShaderTable;
		GfxRayTracingShaderTableDX12 hitGroupShaderTable;
		GfxRayTracingShaderTableDX12 missShaderTable;
	};

	struct GfxRayTracingBindingDX12
	{
		uint32_t bindingIndex;
		uint8_t slotIndex;
	};

	struct GfxRayTracingTextureBindingDX12
	{
		uint32_t bindingIndex;
		uint8_t srvSlot;
		uint8_t samplerSlot;
	};
}

namespace Blueberry
{
	struct GfxRayTracingMaterialBindingDX12
	{
		List<GfxRayTracingTextureBindingDX12> bindlessTextures;
	};

	struct GfxRayTracingBindingStateDX12
	{
		List<GfxRayTracingBindingDX12> cbvs;
		List<GfxRayTracingBindingDX12> textureSrvs;
		List<GfxRayTracingBindingDX12> textureUavs;
		List<GfxRayTracingBindingDX12> samplers;

		Dictionary<ObjectId, GfxRayTracingMaterialBindingDX12> materialBindings;
	};

	class GfxRayTracingRenderStateCacheDX12 : public GfxRenderStateCache
	{
	public:
		GfxRayTracingRenderStateCacheDX12() = default;
		GfxRayTracingRenderStateCacheDX12(GfxDeviceDX12* device);
		~GfxRayTracingRenderStateCacheDX12() = default;

		GfxRayTracingRenderStateDX12 GetRenderState(RayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure);

	private:
		void FillShaderBindingTable(GfxRayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, const GfxRayTracingRenderStateDX12& renderState, GfxRayTracingPipelineStateDX12& pipelineState, GfxRayTracingBindingStateDX12& bindingState);
		void FillRenderState(GfxRayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, GfxRayTracingRenderStateDX12& renderState, const GfxRayTracingPipelineStateDX12& pipelineState, const GfxRayTracingBindingStateDX12& bindingState);
		
	private:
		GfxDeviceDX12* m_Device = nullptr;
		Dictionary<size_t, std::pair<GfxRayTracingPipelineStateDX12, GfxRayTracingBindingStateDX12>> m_PipelineBindingStates;
	};
}