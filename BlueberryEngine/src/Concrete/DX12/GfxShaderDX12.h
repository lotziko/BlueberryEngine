#pragma once

#include "..\..\Blueberry\Graphics\GfxShader.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\VertexLayout.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

namespace Blueberry
{
	template<typename BaseType>
	class GfxShaderDX12 : public BaseType
	{
	protected:
		ByteData m_Blob;
		List<std::pair<size_t, uint8_t>> m_ConstantBufferSlots = {};
		List<std::pair<size_t, std::pair<uint8_t, uint8_t>>> m_TextureSRVSamplerSlots = {};
		List<std::tuple<FilterMode, WrapMode, uint32_t>> m_StaticSamplerSlots = {};
		List<std::pair<size_t, uint8_t>> m_BufferSRVSlots = {};

		friend class GfxDeviceDX12;
		friend class GfxRenderStateCacheDX12;
	};

	class GfxVertexShaderDX12 : public GfxShaderDX12<GfxVertexShader>
	{
	public:
		bool Initialize(ID3D12Device* device, const ByteData& vertexData);

	private:
		List<D3D12_INPUT_ELEMENT_DESC> m_InputElementDescs;
		List<String> m_SemanticNames;
		uint8_t m_LayoutIndices[VERTEX_ATTRIBUTE_COUNT];
		uint32_t m_Crc;

		friend class GfxDeviceDX12;
		friend class GfxRenderStateCacheDX12;
	};

	class GfxGeometryShaderDX12 : public GfxShaderDX12<GfxGeometryShader>
	{
	public:
		bool Initialize(ID3D12Device* device, const ByteData& geometryData);

		friend class GfxDeviceDX12;
		friend class GfxRenderStateCacheDX12;
	};

	class GfxFragmentShaderDX12 : public GfxShaderDX12<GfxFragmentShader>
	{
	public:
		bool Initialize(ID3D12Device* device, const ByteData& fragmentData);

		List<std::pair<size_t, std::pair<uint8_t, uint8_t>>> m_BindlessTextureSRVSamplerSlots = {};

		friend class GfxDeviceDX12;
		friend class GfxRenderStateCacheDX12;
	};
}