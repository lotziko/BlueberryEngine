#include "bbpch.h"
#include "GfxTextureDX11.h"

namespace Blueberry
{
	GfxTextureDX11::GfxTextureDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	GfxTextureDX11::~GfxTextureDX11()
	{
		m_Texture = nullptr;
		m_ResourceView = nullptr;
		m_SamplerState = nullptr;
		m_RenderTargetView = nullptr;
		m_DepthStencilView = nullptr;
	}

	bool IsCompressed(DXGI_FORMAT format)
	{
		switch (format)
		{
		case DXGI_FORMAT_BC1_TYPELESS:
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC1_UNORM_SRGB:
		case DXGI_FORMAT_BC2_TYPELESS:
		case DXGI_FORMAT_BC2_UNORM:
		case DXGI_FORMAT_BC2_UNORM_SRGB:
		case DXGI_FORMAT_BC3_TYPELESS:
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC3_UNORM_SRGB:
		case DXGI_FORMAT_BC4_TYPELESS:
		case DXGI_FORMAT_BC4_UNORM:
		case DXGI_FORMAT_BC4_SNORM:
		case DXGI_FORMAT_BC5_TYPELESS:
		case DXGI_FORMAT_BC5_UNORM:
		case DXGI_FORMAT_BC5_SNORM:
		case DXGI_FORMAT_BC6H_TYPELESS:
		case DXGI_FORMAT_BC6H_UF16:
		case DXGI_FORMAT_BC6H_SF16:
		case DXGI_FORMAT_BC7_TYPELESS:
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
			return true;

		default:
			return false;
		}
	}

	bool GfxTextureDX11::Create(const TextureProperties& properties)
	{
		m_Width = properties.width;
		m_Height = properties.height;

		if (properties.data != nullptr)
		{
			if (IsCompressed((DXGI_FORMAT)properties.format))
			{
				D3D11_SUBRESOURCE_DATA* subresourceDatas = new D3D11_SUBRESOURCE_DATA[properties.mipCount];
				int width = properties.width;
				int height = properties.height;
				char* ptr = (char*)properties.data;
				for (int i = 0; i < properties.mipCount; ++i)
				{
					D3D11_SUBRESOURCE_DATA subresourceData;
					subresourceData.pSysMem = ptr;
					subresourceData.SysMemPitch = 16 * (width / 4);
					subresourceData.SysMemSlicePitch = 0;
					subresourceDatas[i] = subresourceData;

					ptr += subresourceData.SysMemPitch * (height / 4);
					width /= 2;
					height /= 2;
				}
				return Initialize(subresourceDatas, properties);
			}
			else
			{
				D3D11_SUBRESOURCE_DATA subresourceData;

				subresourceData.pSysMem = properties.data;
				subresourceData.SysMemPitch = properties.width * 4;

				return Initialize(&subresourceData, properties);
			}
		}
		else
		{
			return Initialize(nullptr, properties);
		}
	}

	UINT GfxTextureDX11::GetWidth() const
	{
		return m_Width;
	}

	UINT GfxTextureDX11::GetHeight() const
	{
		return m_Height;
	}

	void* GfxTextureDX11::GetHandle()
	{
		return m_ResourceView.Get();
	}

	void GfxTextureDX11::GetData(void* data)
	{
		D3D11_MAPPED_SUBRESOURCE mappedTexture;
		ZeroMemory(&mappedTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));
		HRESULT hr = m_DeviceContext->Map(m_Texture.Get(), 0, D3D11_MAP_READ, 0, &mappedTexture);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get texture data."));
			return;
		}
		memcpy(data, mappedTexture.pData, m_Width * m_Height * sizeof(char) * 4); // TODO handle texture formats
		m_DeviceContext->Unmap(m_Texture.Get(), 0);
	}

	void GfxTextureDX11::SetData(void* data)
	{
	}

	D3D11_TEXTURE_ADDRESS_MODE GetAdressMode(const WrapMode& wrapMode)
	{
		if (wrapMode == WrapMode::Clamp)
		{
			return D3D11_TEXTURE_ADDRESS_CLAMP;
		}
		return D3D11_TEXTURE_ADDRESS_WRAP;
	}

	D3D11_FILTER GetFilter(const FilterMode& filterMode)
	{
		if (filterMode == FilterMode::Linear)
		{
			return D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		}
		return D3D11_FILTER_MIN_MAG_MIP_POINT;
	}

	enum class TextureType
	{
		Resource,
		RenderTarget,
		DepthStencil,
		Staging
	};

	TextureType GetTextureType(const TextureProperties& properties)
	{
		if (properties.format == TextureFormat::D24_UNorm)
		{
			return TextureType::DepthStencil;
		}
		if (properties.isReadable)
		{
			return TextureType::Staging;
		}
		if (properties.isRenderTarget)
		{
			return TextureType::RenderTarget;
		}
		return TextureType::Resource;
	}

	bool GfxTextureDX11::Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, const TextureProperties& properties)
	{
		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

		textureDesc.Width = m_Width;
		textureDesc.Height = m_Height;
		textureDesc.MipLevels = textureDesc.ArraySize = 1;
		textureDesc.Format = (DXGI_FORMAT)properties.format;
		textureDesc.SampleDesc.Count = 1;
		textureDesc.MiscFlags = 0;

		TextureType type = GetTextureType(properties);
		bool compressed = IsCompressed(textureDesc.Format);
		bool generateMipmaps = properties.mipCount > 1 && !compressed;

		switch (type)
		{
		case TextureType::Resource:
			textureDesc.Usage = D3D11_USAGE_DEFAULT;
			textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			if (generateMipmaps)
			{
				textureDesc.MipLevels = 0;
				textureDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
				textureDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
			}
			else if (compressed)
			{
				textureDesc.MipLevels = properties.mipCount;
			}
			break;
		case TextureType::RenderTarget:
			textureDesc.Usage = D3D11_USAGE_DEFAULT;
			textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
			textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			break;
		case TextureType::DepthStencil:
			textureDesc.Usage = D3D11_USAGE_DEFAULT;
			textureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
			textureDesc.CPUAccessFlags = 0;
			break;
		case TextureType::Staging:
			textureDesc.Usage = D3D11_USAGE_STAGING;
			textureDesc.BindFlags = 0;
			textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			break;
		}

		HRESULT hr = m_Device->CreateTexture2D(&textureDesc, generateMipmaps ? nullptr : subresourceData, m_Texture.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
			return false;
		}

		if (type != TextureType::Staging && type != TextureType::DepthStencil)
		{
			if (generateMipmaps)
			{
				D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
				ZeroMemory(&resourceViewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));

				resourceViewDesc.Format = textureDesc.Format;
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				resourceViewDesc.Texture2D.MostDetailedMip = 0;
				resourceViewDesc.Texture2D.MipLevels = -1;

				hr = m_Device->CreateShaderResourceView(m_Texture.Get(), &resourceViewDesc, m_ResourceView.GetAddressOf());
			}
			else
			{
				hr = m_Device->CreateShaderResourceView(m_Texture.Get(), nullptr, m_ResourceView.GetAddressOf());
			}

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
				return false;
			}
			if (generateMipmaps)
			{
				m_DeviceContext->UpdateSubresource(m_Texture.Get(), 0, 0, subresourceData->pSysMem, subresourceData->SysMemPitch, 0);
				m_DeviceContext->GenerateMips(m_ResourceView.Get());
			}

			D3D11_SAMPLER_DESC samplerDesc;
			ZeroMemory(&samplerDesc, sizeof(D3D11_SAMPLER_DESC));

			D3D11_TEXTURE_ADDRESS_MODE adress = GetAdressMode(properties.wrapMode);
			D3D11_FILTER filter = GetFilter(properties.filterMode);

			samplerDesc.Filter = filter;
			samplerDesc.AddressU = adress;
			samplerDesc.AddressV = adress;
			samplerDesc.AddressW = adress;
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
		}

		if (type == TextureType::RenderTarget)
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
			return true;
		}

		if (type == TextureType::DepthStencil)
		{
			D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
			ZeroMemory(&depthStencilViewDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

			depthStencilViewDesc.Format = (DXGI_FORMAT)properties.format;
			depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
			depthStencilViewDesc.Texture2D.MipSlice = 0;

			hr = m_Device->CreateDepthStencilView(m_Texture.Get(), &depthStencilViewDesc, m_DepthStencilView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil view."));
				return false;
			}
		}
		return true;
	}
}