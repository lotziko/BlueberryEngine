#pragma once
#include "Blueberry\Graphics\GfxShader.h"

namespace Blueberry
{
	class GfxShaderDX11 : public GfxShader
	{
	public:
		GfxShaderDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxShaderDX11() final = default;

		bool Compile(const std::wstring& shaderPath);
		bool Initialize(const std::wstring& vertexShaderPath, const std::wstring& pixelShaderPath);

	private:
		bool InitializeVertexLayout();

	private:
		ComPtr<ID3D11VertexShader> m_VertexShader = nullptr;
		ComPtr<ID3D10Blob> m_VertexShaderBuffer = nullptr;

		ComPtr<ID3D11PixelShader> m_PixelShader = nullptr;
		ComPtr<ID3D10Blob> m_PixelShaderBuffer = nullptr;

		ComPtr<ID3D11InputLayout> m_InputLayout;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		std::map<std::size_t, UINT> m_ConstantBufferSlots;
		std::map<std::size_t, UINT> m_TextureSlots;

		friend class GfxDeviceDX11;
	};
}