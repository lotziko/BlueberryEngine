#include "bbpch.h"
#include "DX11Texture.h"

#include "stb\stb_image.h"

namespace Blueberry
{
	DX11Texture::DX11Texture(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	bool DX11Texture::Create(const UINT& width, const UINT& height, bool isRenderTarget)
	{
		m_Width = width;
		m_Height = height;
		return Initialize(nullptr, isRenderTarget);
	}

	bool DX11Texture::Load(const std::string& path)
	{
		stbi_uc* data = nullptr;
		int width, height, channels;

		stbi_set_flip_vertically_on_load(1);
		data = stbi_load(path.c_str(), &width, &height, &channels, 4);

		if (!data)
		{
			std::string errorMsg = "Failed to load texture: " + std::string(path.begin(), path.end());
			BB_ERROR(errorMsg);
			return false;
		}

		m_Width = width;
		m_Height = height;

		D3D11_SUBRESOURCE_DATA* subresourceData;
		subresourceData = new D3D11_SUBRESOURCE_DATA();

		subresourceData->pSysMem = data;
		subresourceData->SysMemPitch = width * 4;

		bool result = Initialize(subresourceData, false);

		stbi_image_free(data);
		delete subresourceData;

		return result;
	}

	UINT DX11Texture::GetWidth() const
	{
		return m_Width;
	}

	UINT DX11Texture::GetHeight() const
	{
		return m_Height;
	}

	void* DX11Texture::GetHandle()
	{
		return m_ResourceView.Get();
	}

	void DX11Texture::BindShaderResource(const UINT& slot)
	{
		m_DeviceContext->PSSetShaderResources(slot, 1, m_ResourceView.GetAddressOf());
		m_DeviceContext->PSSetSamplers(slot, 1, m_SamplerState.GetAddressOf());
	}

	void DX11Texture::BindRenderTarget()
	{
		m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);
	}

	void DX11Texture::Clear(const Color& color)
	{
		m_DeviceContext->ClearRenderTargetView(m_RenderTargetView.Get(), color);
	}

	bool DX11Texture::Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, bool isRenderTarget)
	{
		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

		textureDesc.Width = m_Width;
		textureDesc.Height = m_Height;
		textureDesc.MipLevels = textureDesc.ArraySize = 1;
		textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		textureDesc.SampleDesc.Count = 1;
		textureDesc.Usage = D3D11_USAGE_DEFAULT;
		textureDesc.BindFlags = isRenderTarget ? (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET) : D3D11_BIND_SHADER_RESOURCE;
		textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		textureDesc.MiscFlags = 0;

		HRESULT hr = m_Device->CreateTexture2D(&textureDesc, subresourceData, m_Texture.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
			return false;
		}

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

		if (isRenderTarget)
		{
			D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
			ZeroMemory(&renderTargetViewDesc, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));

			renderTargetViewDesc.Format = textureDesc.Format;
			renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			renderTargetViewDesc.Texture2D.MipSlice = 0;

			hr = m_Device->CreateRenderTargetView(m_Texture.Get(), &renderTargetViewDesc, m_RenderTargetView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
				return false;
			}
		}

		return true;
	}
}