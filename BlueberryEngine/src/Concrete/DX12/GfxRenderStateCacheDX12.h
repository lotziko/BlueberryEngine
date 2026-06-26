#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\GfxRenderStateCache.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	struct GfxTargetInfoDX12
	{
		DXGI_FORMAT renderTargetFormat;
		DXGI_FORMAT depthStencilFormat;
		UINT sampleCount;
		UINT sampleQuality;
	};

	struct GfxPipelineStateKeyDX12
	{
		uint64_t keywordsMask; // global + material
		uint64_t passId;
		ObjectId shaderId;
		uint32_t meshLayoutCrc;
		GfxTargetInfoDX12 targetInfo;
		uint32_t topology;
		uint32_t depthBias;
		float slopeDepthBias;
		bool isCounterClockwise;
		bool isSolid;
		uint16_t padding = 0;

		bool operator==(const GfxPipelineStateKeyDX12& other) const;
		bool operator!=(const GfxPipelineStateKeyDX12& other) const;
	};

	struct GfxRenderStateKeyDX12
	{
		uint64_t keywordsMask; // global + material
		uint64_t passId;
		ObjectId materialId;
		uint32_t padding = 0;

		bool operator==(const GfxRenderStateKeyDX12& other) const;
		bool operator!=(const GfxRenderStateKeyDX12& other) const;
	};
}

template <>
struct std::hash<Blueberry::GfxPipelineStateKeyDX12>
{
	size_t operator()(const Blueberry::GfxPipelineStateKeyDX12& key) const
	{
		return std::hash<uint64_t>()(key.keywordsMask) ^ (std::hash<uint64_t>()(key.passId) << 1) ^ (std::hash<uint32_t>()(key.shaderId) << 2) ^ (std::hash<uint32_t>()(key.meshLayoutCrc) << 3) ^ (std::hash<uint32_t>()(key.topology) << 5);
	}
};

template <>
struct std::hash<Blueberry::GfxRenderStateKeyDX12>
{
	size_t operator()(const Blueberry::GfxRenderStateKeyDX12& key) const
	{
		return std::hash<uint64_t>()(key.keywordsMask) ^ (std::hash<uint64_t>()(key.passId) << 1 ^ (std::hash<uint32_t>()(key.materialId) << 2));
	}
};

namespace Blueberry
{
	class Material;
	class VertexLayout;
	enum class Topology;
	class GfxDeviceDX12;

	struct GfxRenderStateDX12
	{
		ID3D12PipelineState* pipelineState;

		D3D12_CPU_DESCRIPTOR_HANDLE vertexShaderResourceViews[24];
		uint8_t vertexSamplers[16];
		D3D12_CPU_DESCRIPTOR_HANDLE pixelShaderResourceViews[24];
		uint8_t pixelSamplers[16];

		D3D12_CPU_DESCRIPTOR_HANDLE vertexConstantBuffers[8];
		D3D12_CPU_DESCRIPTOR_HANDLE geometryConstantBuffers[8];
		D3D12_CPU_DESCRIPTOR_HANDLE pixelConstantBuffers[8];

		UINT vertexShaderResourceViewsCount;
		UINT vertexSamplersCount;
		UINT pixelShaderResourceViewsCount;
		UINT pixelSamplersCount;
		UINT vertexConstantBuffersCount;
		UINT geometryConstantBuffersCount;
		UINT pixelConstantBuffersCount;

		bool isValid;
	};

	struct GfxPipelineStateDX12
	{
		ID3D12PipelineState* pipelineState;
		uint32_t crc;

		bool isValid;
	};

	struct GfxTextureBindingDX12
	{
		uint32_t bindingIndex;
		bool isGlobal;
		uint8_t srvSlot;
		uint8_t samplerSlot;
	};

	struct GfxBufferBindingDX12
	{
		uint32_t bindingIndex;
		bool isGlobal;
		uint8_t bufferSlot;
		uint8_t srvSlot;
	};

	struct GfxBindingStateDX12
	{
		List<GfxTextureBindingDX12> vertexTextures;
		List<GfxTextureBindingDX12> pixelTextures;

		List<GfxBufferBindingDX12> vertexBuffers;
		List<GfxBufferBindingDX12> geometryBuffers;
		List<GfxBufferBindingDX12> pixelBuffers;

		uint32_t crc;
	};

	class GfxRenderStateCacheDX12 : GfxRenderStateCache
	{
	public:
		GfxRenderStateCacheDX12() = default;
		GfxRenderStateCacheDX12(GfxDeviceDX12* device);
		~GfxRenderStateCacheDX12() = default;

		GfxRenderStateDX12 GetRenderState(Material* material, uint64_t passId, VertexLayout* meshLayout, GfxTargetInfoDX12& targetInfo, Topology topology, uint32_t depthBias, float slopeDepthBias, bool isCounterClockwise, bool isSolid);
	
	private:
		void FillRenderState(Material* material, GfxRenderStateDX12& renderState, const GfxBindingStateDX12& bindingState);

	private:
		GfxDeviceDX12* m_Device;
		Dictionary<GfxPipelineStateKeyDX12, GfxPipelineStateDX12> m_PipelineStates;
		Dictionary<GfxRenderStateKeyDX12, GfxBindingStateDX12> m_BindingStates;
	};
}