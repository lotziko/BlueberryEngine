#pragma once
#include "Blueberry\Graphics\Shader.h"

class DX11Shader : public Shader
{
public:
	DX11Shader(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
	virtual ~DX11Shader() final = default;

	bool Initialize(const std::wstring& vertexShaderPath, const std::wstring& pixelShaderPath);
	virtual void Bind() const override;
	virtual void Unbind() const override;
private:
	ComRef<ID3D11VertexShader> m_VertexShader = nullptr;
	ComRef<ID3D10Blob> m_VertexShaderBuffer = nullptr;

	ComRef<ID3D11PixelShader> m_PixelShader = nullptr;
	ComRef<ID3D10Blob> m_PixelShaderBuffer = nullptr;

	ComRef<ID3D11InputLayout> m_InputLayout;

	ID3D11Device* m_Device;
	ID3D11DeviceContext* m_DeviceContext;
};