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

	uint32_t GetBlockSize(DXGI_FORMAT format)
	{
		switch (format)
		{
		case DXGI_FORMAT_BC1_TYPELESS:
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC1_UNORM_SRGB:
			return 8;
		case DXGI_FORMAT_BC2_TYPELESS:
		case DXGI_FORMAT_BC2_UNORM:
		case DXGI_FORMAT_BC2_UNORM_SRGB:
		case DXGI_FORMAT_BC3_TYPELESS:
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC3_UNORM_SRGB:
			return 16;
		case DXGI_FORMAT_BC4_TYPELESS:
		case DXGI_FORMAT_BC4_UNORM:
		case DXGI_FORMAT_BC4_SNORM:
			return 8;
		case DXGI_FORMAT_BC5_TYPELESS:
		case DXGI_FORMAT_BC5_UNORM:
		case DXGI_FORMAT_BC5_SNORM:
		case DXGI_FORMAT_BC6H_TYPELESS:
		case DXGI_FORMAT_BC6H_UF16:
		case DXGI_FORMAT_BC6H_SF16:
		case DXGI_FORMAT_BC7_TYPELESS:
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
			return 16;

		case DXGI_FORMAT_R8G8B8A8_UNORM:
		case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
			return 4;
		case DXGI_FORMAT_R16G16B16A16_UNORM:
		case DXGI_FORMAT_R16G16B16A16_FLOAT:
			return 8;

		default:
			return 0;
		}
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
			DXGI_FORMAT format = (DXGI_FORMAT)properties.format;
			if (IsCompressed(format))
			{
				uint32_t blockSize = GetBlockSize(format);
				D3D11_SUBRESOURCE_DATA* subresourceDatas = new D3D11_SUBRESOURCE_DATA[properties.mipCount];
				uint32_t width = properties.width;
				uint32_t height = properties.height;
				char* ptr = (char*)properties.data;
				for (uint32_t i = 0; i < properties.mipCount; ++i)
				{
					D3D11_SUBRESOURCE_DATA subresourceData;
					subresourceData.pSysMem = ptr;
					subresourceData.SysMemPitch = std::max(1u, (width + 3) / 4) * blockSize;
					subresourceData.SysMemSlicePitch = 0;
					subresourceDatas[i] = subresourceData;

					ptr += subresourceData.SysMemPitch * std::max(1u, (height + 3) / 4);
					width /= 2;
					height /= 2;
				}
				return InitializeResource(subresourceDatas, properties);
			}
			else
			{
				uint32_t blockSize = properties.dataSize / (properties.width * properties.height);
				D3D11_SUBRESOURCE_DATA subresourceData;

				subresourceData.pSysMem = properties.data;
				subresourceData.SysMemPitch = properties.width * blockSize;

				return InitializeResource(&subresourceData, properties);
			}
		}
		else
		{
			if (properties.isRenderTarget)
			{
				if (properties.format == TextureFormat::D24_UNorm || properties.format == TextureFormat::D32_Float)
				{
					return InitializeDepthStencil(properties);
				}
				else
				{
					return InitializeRenderTarget(properties);
				}
			}
		}
		return false;
	}

	ID3D11Resource* GfxTextureDX11::GetTexture() const
	{
		return m_Texture.Get();
	}

	ID3D11ShaderResourceView* GfxTextureDX11::GetSRV() const
	{
		return m_ResourceView.Get();
	}

	ID3D11RenderTargetView* GfxTextureDX11::GetRTV() const
	{
		return m_RenderTargetView.Get();
	}

	uint32_t GfxTextureDX11::GetWidth() const
	{
		return m_Width;
	}

	uint32_t GfxTextureDX11::GetHeight() const
	{
		return m_Height;
	}

	void* GfxTextureDX11::GetHandle()
	{
		return m_ResourceView.Get();
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
		if (filterMode == FilterMode::CompareDepth)
		{
			return D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
		}
		return D3D11_FILTER_MIN_MAG_MIP_POINT;
	}

	D3D11_COMPARISON_FUNC GetComparison(const FilterMode& filterMode)
	{
		if (filterMode == FilterMode::CompareDepth)
		{
			return D3D11_COMPARISON_LESS;
		}
		return D3D11_COMPARISON_NEVER;
	}

	uint32_t GfxTextureDX11::GetQualityLevel(const DXGI_FORMAT& format, const uint32_t& antiAliasing)
	{
		if (antiAliasing > 1)
		{
			uint32_t qualityLevels;
			HRESULT hr = m_Device->CheckMultisampleQualityLevels(format, antiAliasing, &qualityLevels);
			return qualityLevels - 1;
		}
		return 0;
	}

	bool GfxTextureDX11::InitializeResource(D3D11_SUBRESOURCE_DATA* subresourceData, const TextureProperties& properties)
	{
		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

		textureDesc.Width = m_Width;
		textureDesc.Height = m_Height;
		textureDesc.MipLevels = textureDesc.ArraySize = 1;
		textureDesc.Format = (DXGI_FORMAT)properties.format;
		textureDesc.SampleDesc.Count = 1;
		textureDesc.SampleDesc.Quality = 0;
		textureDesc.MiscFlags = 0;

		bool compressed = IsCompressed(textureDesc.Format);
		bool generateMipmaps = properties.mipCount > 1 && !compressed;

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

		HRESULT hr = m_Device->CreateTexture2D(&textureDesc, generateMipmaps ? nullptr : subresourceData, (ID3D11Texture2D**)m_Texture.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
			return false;
		}

		D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
		ZeroMemory(&resourceViewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));

		resourceViewDesc.Format = textureDesc.Format;
		resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		resourceViewDesc.Texture2D.MostDetailedMip = 0;
		resourceViewDesc.Texture2D.MipLevels = textureDesc.MipLevels;

		if (generateMipmaps)
		{
			resourceViewDesc.Texture2D.MipLevels = -1;
		}

		hr = m_Device->CreateShaderResourceView(m_Texture.Get(), &resourceViewDesc, m_ResourceView.GetAddressOf());
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
		samplerDesc.ComparisonFunc = GetComparison(properties.filterMode);
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

	bool GfxTextureDX11::InitializeRenderTarget(const TextureProperties& properties)
	{
		DXGI_FORMAT format = (DXGI_FORMAT)properties.format;
		uint32_t antiAliasing = std::max(1u, properties.antiAliasing);
		uint32_t arraySize = properties.dimension == TextureDimension::Texture2D ? 1 : properties.depth;

		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

		textureDesc.Width = m_Width;
		textureDesc.Height = m_Height;
		textureDesc.MipLevels = 1;
		textureDesc.ArraySize = arraySize;
		textureDesc.Format = format;
		textureDesc.SampleDesc.Count = antiAliasing;
		textureDesc.SampleDesc.Quality = GetQualityLevel(format, antiAliasing);
		textureDesc.MiscFlags = 0;

		textureDesc.Usage = D3D11_USAGE_DEFAULT;
		textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		textureDesc.CPUAccessFlags = 0;

		HRESULT hr = m_Device->CreateTexture2D(&textureDesc, nullptr, (ID3D11Texture2D**)m_Texture.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
			return false;
		}

		D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
		ZeroMemory(&resourceViewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));

		resourceViewDesc.Format = format;
		if (arraySize == 1)
		{
			resourceViewDesc.ViewDimension = antiAliasing > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
			resourceViewDesc.Texture2D.MostDetailedMip = 0;
			resourceViewDesc.Texture2D.MipLevels = 1;
		}
		else
		{
			if (antiAliasing > 1)
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
				resourceViewDesc.Texture2DMSArray.FirstArraySlice = 0;
				resourceViewDesc.Texture2DMSArray.ArraySize = arraySize;
			}
			else
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
				resourceViewDesc.Texture2DArray.MostDetailedMip = 0;
				resourceViewDesc.Texture2DArray.MipLevels = 1;
				resourceViewDesc.Texture2DArray.FirstArraySlice = 0;
				resourceViewDesc.Texture2DArray.ArraySize = arraySize;
			}
		}

		hr = m_Device->CreateShaderResourceView(m_Texture.Get(), &resourceViewDesc, m_ResourceView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
			return false;
		}

		if (antiAliasing <= 1)
		{
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
			samplerDesc.ComparisonFunc = GetComparison(properties.filterMode);
			samplerDesc.MinLOD = -FLT_MAX;
			samplerDesc.MaxLOD = FLT_MAX;

			hr = m_Device->CreateSamplerState(&samplerDesc, m_SamplerState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create sampler state."));
				return false;
			}
		}

		D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
		ZeroMemory(&renderTargetViewDesc, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));

		renderTargetViewDesc.Format = format;
		if (arraySize == 1)
		{
			renderTargetViewDesc.ViewDimension = antiAliasing > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
			renderTargetViewDesc.Texture2D.MipSlice = 0;
		}
		else
		{
			if (antiAliasing > 1)
			{
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
				renderTargetViewDesc.Texture2DMSArray.FirstArraySlice = 0;
				renderTargetViewDesc.Texture2DMSArray.ArraySize = arraySize;
			}
			else
			{
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
				renderTargetViewDesc.Texture2DArray.MipSlice = 0;
				renderTargetViewDesc.Texture2DArray.FirstArraySlice = 0;
				renderTargetViewDesc.Texture2DArray.ArraySize = arraySize;
			}
		}

		hr = m_Device->CreateRenderTargetView(m_Texture.Get(), &renderTargetViewDesc, m_RenderTargetView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
			return false;
		}

		if (properties.isReadable)
		{
			D3D11_TEXTURE2D_DESC textureDesc;
			ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

			textureDesc.Width = m_Width;
			textureDesc.Height = m_Height;
			textureDesc.MipLevels = 1;
			textureDesc.ArraySize = arraySize;
			textureDesc.Format = format;
			textureDesc.SampleDesc.Count = 1;
			textureDesc.MiscFlags = 0;

			textureDesc.Usage = D3D11_USAGE_STAGING;
			textureDesc.BindFlags = 0;
			textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

			HRESULT hr = m_Device->CreateTexture2D(&textureDesc, nullptr, (ID3D11Texture2D**)m_StagingTexture.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
				return false;
			}
		}

		return true;
	}

	bool GfxTextureDX11::InitializeDepthStencil(const TextureProperties& properties)
	{
		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

		DXGI_FORMAT format = (DXGI_FORMAT)properties.format;
		uint32_t antiAliasing = std::max(1u, properties.antiAliasing);
		uint32_t arraySize = properties.dimension == TextureDimension::Texture2D ? 1 : properties.depth;

		textureDesc.Width = m_Width;
		textureDesc.Height = m_Height;
		textureDesc.MipLevels = 1;
		textureDesc.ArraySize = arraySize;
		textureDesc.Format = format;
		textureDesc.SampleDesc.Count = antiAliasing;
		textureDesc.SampleDesc.Quality = GetQualityLevel(format, antiAliasing);
		textureDesc.MiscFlags = 0;

		switch (format)
		{
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
			textureDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
			break;
		case DXGI_FORMAT_D32_FLOAT:
			textureDesc.Format = DXGI_FORMAT_R32_TYPELESS;
			break;
		}

		textureDesc.Usage = D3D11_USAGE_DEFAULT;
		textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL;
		textureDesc.CPUAccessFlags = 0;

		HRESULT hr = m_Device->CreateTexture2D(&textureDesc, nullptr, (ID3D11Texture2D**)m_Texture.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
			return false;
		}

		D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
		ZeroMemory(&resourceViewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));

		resourceViewDesc.Format = format;
		if (arraySize == 1)
		{
			resourceViewDesc.ViewDimension = antiAliasing > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
			resourceViewDesc.Texture2D.MostDetailedMip = 0;
			resourceViewDesc.Texture2D.MipLevels = textureDesc.MipLevels;
		}
		else
		{
			if (antiAliasing > 1)
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
				resourceViewDesc.Texture2DMSArray.FirstArraySlice = 0;
				resourceViewDesc.Texture2DMSArray.ArraySize = arraySize;
			}
			else
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
				resourceViewDesc.Texture2DArray.MostDetailedMip = 0;
				resourceViewDesc.Texture2DArray.MipLevels = 1;
				resourceViewDesc.Texture2DArray.FirstArraySlice = 0;
				resourceViewDesc.Texture2DArray.ArraySize = arraySize;
			}
		}

		switch (format)
		{
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
			resourceViewDesc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
			break;
		case DXGI_FORMAT_D32_FLOAT:
			resourceViewDesc.Format = DXGI_FORMAT_R32_FLOAT;
			break;
		}

		hr = m_Device->CreateShaderResourceView(m_Texture.Get(), &resourceViewDesc, m_ResourceView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
			return false;
		}

		if (antiAliasing <= 1)
		{
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
			samplerDesc.ComparisonFunc = GetComparison(properties.filterMode);
			samplerDesc.MinLOD = -FLT_MAX;
			samplerDesc.MaxLOD = FLT_MAX;

			hr = m_Device->CreateSamplerState(&samplerDesc, m_SamplerState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create sampler state."));
				return false;
			}
		}

		D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
		ZeroMemory(&depthStencilViewDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

		depthStencilViewDesc.Format = format;
		if (arraySize == 1)
		{
			depthStencilViewDesc.ViewDimension = antiAliasing > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
			depthStencilViewDesc.Texture2D.MipSlice = 0;
		}
		else
		{
			if (antiAliasing > 1)
			{
				depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
				depthStencilViewDesc.Texture2DMSArray.FirstArraySlice = 0;
				depthStencilViewDesc.Texture2DMSArray.ArraySize = arraySize;
			}
			else
			{
				depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
				depthStencilViewDesc.Texture2DArray.MipSlice = 0;
				depthStencilViewDesc.Texture2DArray.FirstArraySlice = 0;
				depthStencilViewDesc.Texture2DArray.ArraySize = arraySize;
			}
		}

		hr = m_Device->CreateDepthStencilView(m_Texture.Get(), &depthStencilViewDesc, m_DepthStencilView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil view."));
			return false;
		}

		return true;
	}
}