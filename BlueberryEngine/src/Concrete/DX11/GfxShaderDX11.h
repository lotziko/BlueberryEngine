#pragma once

#include "..\..\Blueberry\Graphics\GfxShader.h"
#include "Blueberry\Graphics\VertexLayout.h"
#include "..\Windows\ComPtr.h"
#include "DX11.h"

namespace Blueberry
{
	template<typename BaseType, typename ShaderType>
	class GfxShaderDX11 : public BaseType
	{
	protected:
		ComPtr<ShaderType> m_Shader = nullptr;
		ComPtr<ID3DBlob> m_ShaderBuffer = nullptr;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxVertexShaderDX11 : GfxShaderDX11<GfxVertexShader, ID3D11VertexShader>
	{
	public:
		bool Initialize(ID3D11Device* device, void* vertexData);

	private:
		ID3D11InputLayout* CreateLayout();

		ID3D11Device* m_Device;
		List<D3D11_INPUT_ELEMENT_DESC> m_InputElementDescs;
		uint8_t m_LayoutIndices[VERTEX_ATTRIBUTE_COUNT];
		uint32_t m_Crc;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
		friend class GfxInputLayoutCacheDX11;
	};

	class GfxGeometryShaderDX11 : GfxShaderDX11<GfxGeometryShader, ID3D11GeometryShader>
	{
	public:
		bool Initialize(ID3D11Device* device, void* geometryData);

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxFragmentShaderDX11 : GfxShaderDX11<GfxFragmentShader, ID3D11PixelShader>
	{
		bool Initialize(ID3D11Device* device, void* fragmentData);

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};
}