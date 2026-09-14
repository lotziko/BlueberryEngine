#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxDeviceDX12;
	class GfxComputeShader;
	class ComputeShader;

	struct GfxComputeRenderStateDX12
	{
		ID3D12PipelineState* pipelineState;

		D3D12_CPU_DESCRIPTOR_HANDLE constantBuffers[14];
		D3D12_CPU_DESCRIPTOR_HANDLE shaderResourceViews[16];
		D3D12_CPU_DESCRIPTOR_HANDLE unorderedAccessViews[8];
		uint8_t samplers[16];

		UINT constantBuffersCount;
		UINT shaderResourceViewsCount;
		UINT unorderedAccessViewsCount;
		UINT samplersCount;

		bool isValid;
	};

	struct GfxComputePipelineStateDX12
	{
		ComPtr<ID3D12PipelineState> pipelineState;
	};

	struct GfxComputeBindingDX12
	{
		uint32_t bindingIndex;
		uint8_t slotIndex;
	};

	struct GfxComputeStaticSamplerBindingDX12
	{
		uint8_t sampler;
		uint8_t slotIndex;
	};

	struct GfxComputeBindingStateDX12
	{
		List<GfxComputeBindingDX12> cbvs;
		List<GfxComputeBindingDX12> bufferSrvs;
		List<GfxComputeBindingDX12> textureSrvs;
		List<GfxComputeBindingDX12> bufferUavs;
		List<GfxComputeBindingDX12> textureUavs;
		List<GfxComputeBindingDX12> samplers;
		List<GfxComputeStaticSamplerBindingDX12> staticSamplers;
	};

	class GfxComputeRenderStateCacheDX12
	{
	public:
		GfxComputeRenderStateCacheDX12() = default;
		GfxComputeRenderStateCacheDX12(GfxDeviceDX12* device);
		~GfxComputeRenderStateCacheDX12() = default;

		GfxComputeRenderStateDX12 GetRenderState(ComputeShader* shader, uint32_t kernelIndex);

	private:
		void FillRenderState(GfxComputeRenderStateDX12& renderState, const GfxComputePipelineStateDX12& pipelineState, const GfxComputeBindingStateDX12& bindingState);

	private:
		GfxDeviceDX12* m_Device = nullptr;
		Dictionary<size_t, std::pair<GfxComputePipelineStateDX12, GfxComputeBindingStateDX12>> m_PipelineBindingStates;
	};
}