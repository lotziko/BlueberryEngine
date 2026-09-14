#pragma once

#include "..\..\Blueberry\Graphics\GfxShader.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\VertexLayout.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	template<typename BaseType, typename ShaderType>
	class GfxShaderDX11 : public BaseType
	{
	protected:
		ComPtr<ShaderType> m_Shader = nullptr;
		List<std::pair<size_t, uint8_t>> m_ConstantBufferSlots = {};
		List<std::pair<size_t, std::pair<uint8_t, uint8_t>>> m_TextureSRVSamplerSlots = {};
		List<std::tuple<FilterMode, WrapMode, uint32_t>> m_StaticSamplerSlots = {};
		List<std::pair<size_t, uint8_t>> m_BufferSRVSlots = {};

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxVertexShaderDX11 : public GfxShaderDX11<GfxVertexShader, ID3D11VertexShader>
	{
	public:
		bool Initialize(ID3D11Device* device, const ByteData& vertexData);

	private:
		ID3D11InputLayout* CreateLayout();

		ID3D11Device* m_Device;
		List<D3D11_INPUT_ELEMENT_DESC> m_InputElementDescs;
		List<String> m_SemanticNames;
		ByteData m_Blob;
		uint8_t m_LayoutIndices[VERTEX_ATTRIBUTE_COUNT];
		uint32_t m_Crc;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxGeometryShaderDX11 : public GfxShaderDX11<GfxGeometryShader, ID3D11GeometryShader>
	{
	public:
		bool Initialize(ID3D11Device* device, const ByteData& geometryData);

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxFragmentShaderDX11 : public GfxShaderDX11<GfxFragmentShader, ID3D11PixelShader>
	{
	public:
		bool Initialize(ID3D11Device* device, const ByteData& fragmentData);

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};
}