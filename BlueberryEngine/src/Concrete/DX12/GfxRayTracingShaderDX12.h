#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\GfxRayTracingShader.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxRayTracingShaderDX12 : public GfxRayTracingShader
	{
	public:
		GfxRayTracingShaderDX12() = default;
		virtual ~GfxRayTracingShaderDX12() = default;

		bool Initialize(ID3D12Device* device, const ByteData& rayTracingData);

	private:
		ByteData m_Blob;

		uint32_t m_AccelerationStructureSlot = 0;
		List<std::pair<size_t, uint32_t>> m_ConstantBufferSlots = {};
		List<std::pair<size_t, uint32_t>> m_TextureSRVSlots = {};
		List<std::pair<size_t, uint32_t>> m_TextureUAVSlots = {};
		List<std::pair<size_t, uint32_t>> m_SamplerSlots = {};

		List<std::pair<size_t, std::pair<uint8_t, uint8_t>>> m_BindlessTextureSRVSamplerSlots = {};

		friend class GfxRayTracingRenderStateCacheDX12;
	};
}