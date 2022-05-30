#include "bbpch.h"
#include "DX11Texture.h"

#include "stb\stb_image.h"

DX11Texture::DX11Texture(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
{
}

bool DX11Texture::Initialize(const std::string& path)
{
	stbi_uc* data = nullptr;
	int width, height, channels;
	data = stbi_load(path.c_str(), &width, &height, &channels, 4);

	if (!data)
	{
		std::string errorMsg = "Failed to load texture: " + std::string(path.begin(), path.end());
		BB_ERROR(errorMsg);
		return false;
	}

	m_Width = width;
	m_Height = height;

	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
	
	textureDesc.Width = width;
	textureDesc.Height = height;
	textureDesc.MipLevels = textureDesc.ArraySize = 1;
	textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.Usage = D3D11_USAGE_DYNAMIC;
	textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	textureDesc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA subresourceData;
	ZeroMemory(&subresourceData, sizeof(D3D11_MAPPED_SUBRESOURCE));

	subresourceData.pSysMem = data;
	subresourceData.SysMemPitch = width * 4;

	HRESULT hr = m_Device->CreateTexture2D(&textureDesc, &subresourceData, m_Texture.GetAddressOf());
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
		return false;
	}

	stbi_image_free(data);

	hr = m_Device->CreateShaderResourceView(m_Texture.Get(), nullptr, m_ResourceView.GetAddressOf());
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
		return false;
	}

	D3D11_SAMPLER_DESC samplerDesc;
	ZeroMemory(&samplerDesc, sizeof(D3D11_SAMPLER_DESC));

	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samplerDesc.MinLOD = -FLT_MAX;
	samplerDesc.MaxLOD = FLT_MAX;

	hr = m_Device->CreateSamplerState(&samplerDesc, m_SamplerState.GetAddressOf());
	if (FAILED(hr))
	{
		BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create sampler state."));
		return false;
	}

	return true;
}

void* DX11Texture::GetHandle()
{
	return m_ResourceView.Get();
}

void DX11Texture::Bind() const
{
	m_DeviceContext->PSSetShaderResources(0, 1, m_ResourceView.GetAddressOf());
	m_DeviceContext->PSSetSamplers(0, 1, m_SamplerState.GetAddressOf());
}
