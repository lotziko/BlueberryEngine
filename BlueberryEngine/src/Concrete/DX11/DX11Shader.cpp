#include "bbpch.h"
#include "DX11Shader.h"
#include "DX11GraphicsDevice.h"

namespace Blueberry
{
	DX11Shader::DX11Shader(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	bool DX11Shader::Compile(const std::wstring& shaderPath)
	{
		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

		HRESULT hr = D3DCompileFromFile(shaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "Vertex", "vs_5_0", flags, 0, m_VertexShaderBuffer.GetAddressOf(), nullptr);
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to compile vertex shader: " + std::string(shaderPath.begin(), shaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}
	
		hr = m_Device->CreateVertexShader(m_VertexShaderBuffer->GetBufferPointer(), m_VertexShaderBuffer->GetBufferSize(), NULL, m_VertexShader.GetAddressOf());
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to create vertex shader: " + std::string(shaderPath.begin(), shaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}

		hr = D3DCompileFromFile(shaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "Fragment", "ps_5_0", flags, 0, m_PixelShaderBuffer.GetAddressOf(), nullptr);
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to compile fragment shader: " + std::string(shaderPath.begin(), shaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}

		hr = m_Device->CreatePixelShader(m_PixelShaderBuffer->GetBufferPointer(), m_PixelShaderBuffer->GetBufferSize(), NULL, m_PixelShader.GetAddressOf());
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to create pixel shader: " + std::string(shaderPath.begin(), shaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}

		if (!InitializeVertexLayout())
		{
			return false;
		}

		return true;
	}

	bool DX11Shader::Initialize(const std::wstring& vertexShaderPath, const std::wstring& pixelShaderPath)
	{
		HRESULT hr = D3DReadFileToBlob(vertexShaderPath.c_str(), m_VertexShaderBuffer.GetAddressOf());
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to load vertex shader: " + std::string(vertexShaderPath.begin(), vertexShaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}

		hr = m_Device->CreateVertexShader(m_VertexShaderBuffer->GetBufferPointer(), m_VertexShaderBuffer->GetBufferSize(), NULL, m_VertexShader.GetAddressOf());
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to create vertex shader: " + std::string(vertexShaderPath.begin(), vertexShaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}

		hr = D3DReadFileToBlob(pixelShaderPath.c_str(), m_PixelShaderBuffer.GetAddressOf());
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to load pixel shader: " + std::string(pixelShaderPath.begin(), pixelShaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}

		hr = m_Device->CreatePixelShader(m_PixelShaderBuffer->GetBufferPointer(), m_PixelShaderBuffer->GetBufferSize(), NULL, m_PixelShader.GetAddressOf());
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to create pixel shader: " + std::string(pixelShaderPath.begin(), pixelShaderPath.end());
			BB_ERROR(errorMsg);
			return false;
		}

		if (!InitializeVertexLayout())
		{
			return false;
		}

		return true;
	}

	void DX11Shader::Bind() const
	{
		m_DeviceContext->IASetInputLayout(m_InputLayout.Get());
		m_DeviceContext->VSSetShader(m_VertexShader.Get(), NULL, 0);
		m_DeviceContext->PSSetShader(m_PixelShader.Get(), NULL, 0);

		m_DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY::D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	}

	void DX11Shader::Unbind() const
	{
	}

	bool DX11Shader::InitializeVertexLayout()
	{
		ID3D11ShaderReflection* vertexShaderReflection;
		HRESULT hr = D3DReflect(m_VertexShaderBuffer->GetBufferPointer(), m_VertexShaderBuffer->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&vertexShaderReflection);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get vertex shader reflection."));
			return false;
		}

		D3D11_SHADER_DESC shaderDesc;
		vertexShaderReflection->GetDesc(&shaderDesc);

		std::vector<D3D11_INPUT_ELEMENT_DESC> inputElementDescArray;
		unsigned int parameterCount = shaderDesc.InputParameters;

		for (unsigned int i = 0; i < parameterCount; ++i)
		{
			D3D11_SIGNATURE_PARAMETER_DESC paramDesc;
			vertexShaderReflection->GetInputParameterDesc(i, &paramDesc);

			D3D11_INPUT_ELEMENT_DESC inputElementDesc;

			inputElementDesc.SemanticName = paramDesc.SemanticName;
			inputElementDesc.SemanticIndex = paramDesc.SemanticIndex;

			if (paramDesc.Mask == 1) {
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32_FLOAT;
			}
			else if (paramDesc.Mask <= 3) {
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
			}
			else if (paramDesc.Mask <= 7) {
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
			}
			else if (paramDesc.Mask <= 15) {
				if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32A32_SINT;
				else if (paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) inputElementDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			}

			inputElementDesc.InputSlot = 0;
			inputElementDesc.AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
			inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
			inputElementDesc.InstanceDataStepRate = 0;

			inputElementDescArray.push_back(inputElementDesc);
		}

		hr = m_Device->CreateInputLayout(&inputElementDescArray[0], parameterCount, m_VertexShaderBuffer->GetBufferPointer(), m_VertexShaderBuffer->GetBufferSize(), m_InputLayout.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating input layout."));
			return false;
		}

		return true;
	}
}