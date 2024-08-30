#pragma once
#include "Blueberry\Graphics\GfxShader.h"

namespace Blueberry
{
	template<typename BaseType, typename ShaderType>
	class GfxShaderDX11 : public BaseType
	{
	protected:
		ComPtr<ShaderType> m_Shader = nullptr;
		ComPtr<ID3DBlob> m_ShaderBuffer = nullptr;

		friend class GfxDeviceDX11;
	};

	class GfxVertexShaderDX11 : GfxShaderDX11<GfxVertexShader, ID3D11VertexShader>
	{
	public:
		bool Initialize(ID3D11Device* device, void* vertexData);

	private:
		ComPtr<ID3D11InputLayout> m_InputLayout;

		friend class GfxDeviceDX11;
	};

	class GfxGeometryShaderDX11 : GfxShaderDX11<GfxGeometryShader, ID3D11GeometryShader>
	{
	public:
		bool Initialize(ID3D11Device* device, void* geometryData);

		friend class GfxDeviceDX11;
	};

	class GfxFragmentShaderDX11 : GfxShaderDX11<GfxFragmentShader, ID3D11PixelShader>
	{
		bool Initialize(ID3D11Device* device, void* fragmentData);

		friend class GfxDeviceDX11;
	};
}