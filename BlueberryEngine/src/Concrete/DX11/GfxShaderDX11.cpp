#include "bbpch.h"
#include "GfxShaderDX11.h"
#include "GfxDeviceDX11.h"

namespace Blueberry
{
	bool GfxVertexShaderDX11::Initialize(ID3D11Device* device, void* vertexData)
	{
		if (vertexData == nullptr)
		{
			BB_ERROR("Vertex data is empty.");
			return false;
		}

		m_ShaderBuffer.Attach((ID3DBlob*)vertexData);
		HRESULT hr = device->CreateVertexShader(m_ShaderBuffer->GetBufferPointer(), m_ShaderBuffer->GetBufferSize(), NULL, m_Shader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create vertex shader from data.");
			return false;
		}

		ID3D11ShaderReflection* vertexShaderReflection;
		hr = D3DReflect(m_ShaderBuffer->GetBufferPointer(), m_ShaderBuffer->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&vertexShaderReflection);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get vertex shader reflection."));
			return false;
		}

		// Slots
		D3D11_SHADER_DESC vertexShaderDesc;
		vertexShaderReflection->GetDesc(&vertexShaderDesc);

		UINT constantBufferCount = vertexShaderDesc.ConstantBuffers;

		for (UINT i = 0; i < constantBufferCount; i++)
		{
			ID3D11ShaderReflectionConstantBuffer* constantBufferReflection = vertexShaderReflection->GetConstantBufferByIndex(i);
			D3D11_SHADER_BUFFER_DESC shaderBufferDesc;
			constantBufferReflection->GetDesc(&shaderBufferDesc);
			if (shaderBufferDesc.Type == D3D_CT_CBUFFER)
			{
				m_ConstantBufferSlots.insert({ TO_HASH(std::string(shaderBufferDesc.Name)), i });
			}
			else if (shaderBufferDesc.Type == D3D_CT_RESOURCE_BIND_INFO)
			{
				m_StructuredBufferSlots.insert({ TO_HASH(std::string(shaderBufferDesc.Name)), std::make_pair(i, 0) });
			}
		}

		unsigned int resourceBindingCount = vertexShaderDesc.BoundResources;

		for (UINT i = 0; i < resourceBindingCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			vertexShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			if (inputBindDesc.Type == D3D_SIT_TEXTURE)
			{
				// TODO
			}
			else if (inputBindDesc.Type == D3D_SIT_STRUCTURED)
			{
				auto pair = &m_StructuredBufferSlots[TO_HASH(std::string(inputBindDesc.Name))];
				pair->second = i;
			}
		}

		// Input layout
		std::vector<D3D11_INPUT_ELEMENT_DESC> inputElementDescArray;
		UINT parameterCount = vertexShaderDesc.InputParameters;

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
				
			if (strcmp(paramDesc.SemanticName, "RenderInstance") == 0)
			{
				inputElementDesc.InputSlot = 1;
				inputElementDesc.AlignedByteOffset = 0;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_INSTANCE_DATA;
				inputElementDesc.InstanceDataStepRate = 1;
			}
			else
			{
				inputElementDesc.InputSlot = 0;
				inputElementDesc.AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
				inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
				inputElementDesc.InstanceDataStepRate = 0;
			}
			inputElementDescArray.push_back(inputElementDesc);
		}

		hr = device->CreateInputLayout(&inputElementDescArray[0], inputElementDescArray.size(), m_ShaderBuffer->GetBufferPointer(), m_ShaderBuffer->GetBufferSize(), m_InputLayout.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating input layout."));
			return false;
		}

		return true;
	}

	bool GfxGeometryShaderDX11::Initialize(ID3D11Device* device, void* geometryData)
	{
		if (geometryData == nullptr)
		{
			BB_ERROR("Geometry data is empty.");
			return false;
		}

		m_ShaderBuffer.Attach((ID3DBlob*)geometryData);
		HRESULT hr = device->CreateGeometryShader(m_ShaderBuffer->GetBufferPointer(), m_ShaderBuffer->GetBufferSize(), NULL, m_Shader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create geometry shader from data.");
			return false;
		}

		// Slots
		ID3D11ShaderReflection* geometryShaderReflection;
		hr = D3DReflect(m_ShaderBuffer->GetBufferPointer(), m_ShaderBuffer->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&geometryShaderReflection);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get geometry shader reflection."));
			return false;
		}

		D3D11_SHADER_DESC geometryShaderDesc;
		geometryShaderReflection->GetDesc(&geometryShaderDesc);

		UINT constantBufferCount = geometryShaderDesc.ConstantBuffers;

		for (UINT i = 0; i < constantBufferCount; i++)
		{
			ID3D11ShaderReflectionConstantBuffer* constantBufferReflection = geometryShaderReflection->GetConstantBufferByIndex(i);
			D3D11_SHADER_BUFFER_DESC shaderBufferDesc;
			constantBufferReflection->GetDesc(&shaderBufferDesc);
			m_ConstantBufferSlots.insert({ TO_HASH(std::string(shaderBufferDesc.Name)), i });
		}

		return true;
	}

	bool GfxFragmentShaderDX11::Initialize(ID3D11Device* device, void* fragmentData)
	{
		if (fragmentData == nullptr)
		{
			BB_ERROR("Fragment data is empty.");
			return false;
		}

		m_ShaderBuffer.Attach((ID3DBlob*)fragmentData);
		HRESULT hr = device->CreatePixelShader(m_ShaderBuffer->GetBufferPointer(), m_ShaderBuffer->GetBufferSize(), NULL, m_Shader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create fragment shader from data.");
			return false;
		}

		// Slots
		ID3D11ShaderReflection* pixelShaderReflection;
		hr = D3DReflect(m_ShaderBuffer->GetBufferPointer(), m_ShaderBuffer->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&pixelShaderReflection);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get pixel shader reflection."));
			return false;
		}

		D3D11_SHADER_DESC pixelShaderDesc;
		pixelShaderReflection->GetDesc(&pixelShaderDesc);

		UINT constantBufferCount = pixelShaderDesc.ConstantBuffers;

		for (UINT i = 0; i < constantBufferCount; i++)
		{
			ID3D11ShaderReflectionConstantBuffer* constantBufferReflection = pixelShaderReflection->GetConstantBufferByIndex(i);
			D3D11_SHADER_BUFFER_DESC shaderBufferDesc;
			constantBufferReflection->GetDesc(&shaderBufferDesc);
			m_ConstantBufferSlots.insert({ TO_HASH(std::string(shaderBufferDesc.Name)), i });
		}

		unsigned int resourceBindingCount = pixelShaderDesc.BoundResources;

		for (UINT i = 0; i < resourceBindingCount; i++)
		{
			D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
			pixelShaderReflection->GetResourceBindingDesc(i, &inputBindDesc);
			if (inputBindDesc.Type == D3D_SIT_TEXTURE)
			{
				UINT samplerSlot = -1;
				if (inputBindDesc.NumSamples > 8)
				{
					for (UINT j = 0; j < resourceBindingCount; j++)
					{
						D3D11_SHADER_INPUT_BIND_DESC samplerInputBindDesc;
						pixelShaderReflection->GetResourceBindingDesc(j, &samplerInputBindDesc);
						if (samplerInputBindDesc.Type == D3D10_SIT_SAMPLER)
						{
							if (strncmp(samplerInputBindDesc.Name, inputBindDesc.Name, strlen(inputBindDesc.Name)) == 0)
							{
								samplerSlot = samplerInputBindDesc.BindPoint;
								break;
							}
						}
					}
				}
				m_TextureSlots.insert({ TO_HASH(std::string(inputBindDesc.Name)), std::make_pair(inputBindDesc.BindPoint, samplerSlot) });
			}
			else if (inputBindDesc.Type == D3D_SIT_STRUCTURED)
			{

			}
		}

		return true;
	}
}