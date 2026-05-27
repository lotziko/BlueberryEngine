#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\GfxComputeShader.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	class GfxComputeShaderDX12 : public GfxComputeShader
	{
	public:
		GfxComputeShaderDX12() = default;
		virtual ~GfxComputeShaderDX12() = default;

		bool Initialize(ID3D12Device* device, const ByteData& computeData);

	private:
		ByteData m_Blob;
		List<std::pair<size_t, uint32_t>> m_ConstantBufferSlots = {};
		List<std::pair<size_t, uint32_t>> m_TextureSRVSlots = {};
		List<std::pair<size_t, uint32_t>> m_BufferSRVSlots = {};
		List<std::pair<size_t, uint32_t>> m_TextureUAVSlots = {};
		List<std::pair<size_t, uint32_t>> m_BufferUAVSlots = {};
		List<std::pair<size_t, uint32_t>> m_SamplerSlots = {};

		friend class GfxDeviceDX12;
		friend class GfxComputeRenderStateCacheDX12;
	};
}