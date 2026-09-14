#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	class GfxDeviceDX11;
	class GfxComputeShader;
	class ComputeShader;

	struct GfxComputeRenderStateDX11
	{
		ID3D11ComputeShader* computeShader;

		ID3D11Buffer* constantBuffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
		ID3D11ShaderResourceView* shaderResourceViews[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 8];
		ID3D11UnorderedAccessView* unorderedAccessViews[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 16];
		ID3D11SamplerState* samplerStates[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];

		UINT constantBuffersCount;
		UINT shaderResourceViewsCount;
		UINT unorderedAccessViewsCount;
		UINT samplerStatesCount;

		bool isValid;
	};

	struct GfxComputePipelineStateDX11
	{
		ID3D11ComputeShader* computeShader;
	};

	struct GfxComputeBindingDX11
	{
		uint32_t bindingIndex;
		uint8_t slotIndex;
	};

	struct GfxComputeStaticSamplerBindingDX11
	{
		ID3D11SamplerState* samplerState;
		uint8_t slotIndex;
	};

	struct GfxComputeBindingStateDX11
	{
		List<GfxComputeBindingDX11> cbvs;
		List<GfxComputeBindingDX11> bufferSrvs;
		List<GfxComputeBindingDX11> textureSrvs;
		List<GfxComputeBindingDX11> bufferUavs;
		List<GfxComputeBindingDX11> textureUavs;
		List<GfxComputeBindingDX11> samplers;
		List<GfxComputeStaticSamplerBindingDX11> staticSamplers;
	};

	class GfxComputeRenderStateCacheDX11
	{
	public:
		GfxComputeRenderStateCacheDX11() = default;
		GfxComputeRenderStateCacheDX11(GfxDeviceDX11* device);
		~GfxComputeRenderStateCacheDX11() = default;

		GfxComputeRenderStateDX11 GetRenderState(ComputeShader* shader, uint32_t kernelIndex);

	private:
		void FillRenderState(GfxComputeShader* shader, GfxComputeRenderStateDX11& renderState, const GfxComputePipelineStateDX11& pipelineState, const GfxComputeBindingStateDX11& bindingState);

	private:
		GfxDeviceDX11* m_Device = nullptr;
		Dictionary<size_t, std::pair<GfxComputePipelineStateDX11, GfxComputeBindingStateDX11>> m_PipelineBindingStates;
	};
}