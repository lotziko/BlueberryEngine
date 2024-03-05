#include "bbpch.h"
#include "GfxComputeShaderDX11.h"

namespace Blueberry
{
	GfxComputeShaderDX11::GfxComputeShaderDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	bool GfxComputeShaderDX11::Initialize(void* computeData)
	{
		if (computeData == nullptr)
		{
			BB_ERROR("Compute data is empty.");
			return false;
		}

		m_ComputeShaderBuffer.Attach((ID3DBlob*)computeData);
		HRESULT hr = m_Device->CreateComputeShader(m_ComputeShaderBuffer->GetBufferPointer(), m_ComputeShaderBuffer->GetBufferSize(), NULL, m_ComputeShader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to create compute shader from data.");
			return false;
		}

		return false;
	}
}