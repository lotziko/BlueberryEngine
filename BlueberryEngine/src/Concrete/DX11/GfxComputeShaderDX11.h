#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\GfxComputeShader.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	class GfxComputeShaderDX11 : public GfxComputeShader
	{
	public:
		GfxComputeShaderDX11() = default;
		virtual ~GfxComputeShaderDX11() = default;

		bool Initialize(ID3D11Device* device, const ByteData& computeData);

	private:
		ComPtr<ID3D11ComputeShader> m_ComputeShader = nullptr;

		List<std::pair<size_t, uint32_t>> m_ConstantBufferSlots = {};
		List<std::pair<size_t, uint32_t>> m_TextureSRVSlots = {};
		List<std::pair<size_t, uint32_t>> m_BufferSRVSlots = {};
		List<std::pair<size_t, uint32_t>> m_TextureUAVSlots = {};
		List<std::pair<size_t, uint32_t>> m_BufferUAVSlots = {};
		List<std::pair<size_t, uint32_t>> m_SamplerSlots = {};

		friend class GfxDeviceDX11;
		friend class GfxComputeRenderStateCacheDX11;
	};
}