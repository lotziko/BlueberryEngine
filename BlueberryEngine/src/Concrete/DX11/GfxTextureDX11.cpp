#include "GfxTextureDX11.h"

#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	GfxTextureDX11::GfxTextureDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	uint32_t GetBitsPerPixel(DXGI_FORMAT format) noexcept
	{
		switch (static_cast<int>(format))
		{
		case DXGI_FORMAT_R32G32B32A32_TYPELESS:
		case DXGI_FORMAT_R32G32B32A32_FLOAT:
		case DXGI_FORMAT_R32G32B32A32_UINT:
		case DXGI_FORMAT_R32G32B32A32_SINT:
			return 128;

		case DXGI_FORMAT_R32G32B32_TYPELESS:
		case DXGI_FORMAT_R32G32B32_FLOAT:
		case DXGI_FORMAT_R32G32B32_UINT:
		case DXGI_FORMAT_R32G32B32_SINT:
			return 96;

		case DXGI_FORMAT_R16G16B16A16_TYPELESS:
		case DXGI_FORMAT_R16G16B16A16_FLOAT:
		case DXGI_FORMAT_R16G16B16A16_UNORM:
		case DXGI_FORMAT_R16G16B16A16_UINT:
		case DXGI_FORMAT_R16G16B16A16_SNORM:
		case DXGI_FORMAT_R16G16B16A16_SINT:
		case DXGI_FORMAT_R32G32_TYPELESS:
		case DXGI_FORMAT_R32G32_FLOAT:
		case DXGI_FORMAT_R32G32_UINT:
		case DXGI_FORMAT_R32G32_SINT:
		case DXGI_FORMAT_R32G8X24_TYPELESS:
		case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
		case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
		case DXGI_FORMAT_Y416:
		case DXGI_FORMAT_Y210:
		case DXGI_FORMAT_Y216:
			return 64;

		case DXGI_FORMAT_R10G10B10A2_TYPELESS:
		case DXGI_FORMAT_R10G10B10A2_UNORM:
		case DXGI_FORMAT_R10G10B10A2_UINT:
		case DXGI_FORMAT_R11G11B10_FLOAT:
		case DXGI_FORMAT_R8G8B8A8_TYPELESS:
		case DXGI_FORMAT_R8G8B8A8_UNORM:
		case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		case DXGI_FORMAT_R8G8B8A8_UINT:
		case DXGI_FORMAT_R8G8B8A8_SNORM:
		case DXGI_FORMAT_R8G8B8A8_SINT:
		case DXGI_FORMAT_R16G16_TYPELESS:
		case DXGI_FORMAT_R16G16_FLOAT:
		case DXGI_FORMAT_R16G16_UNORM:
		case DXGI_FORMAT_R16G16_UINT:
		case DXGI_FORMAT_R16G16_SNORM:
		case DXGI_FORMAT_R16G16_SINT:
		case DXGI_FORMAT_R32_TYPELESS:
		case DXGI_FORMAT_D32_FLOAT:
		case DXGI_FORMAT_R32_FLOAT:
		case DXGI_FORMAT_R32_UINT:
		case DXGI_FORMAT_R32_SINT:
		case DXGI_FORMAT_R24G8_TYPELESS:
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
		case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
		case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
		case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
		case DXGI_FORMAT_R8G8_B8G8_UNORM:
		case DXGI_FORMAT_G8R8_G8B8_UNORM:
		case DXGI_FORMAT_B8G8R8A8_UNORM:
		case DXGI_FORMAT_B8G8R8X8_UNORM:
		case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
		case DXGI_FORMAT_B8G8R8A8_TYPELESS:
		case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
		case DXGI_FORMAT_B8G8R8X8_TYPELESS:
		case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
		case DXGI_FORMAT_AYUV:
		case DXGI_FORMAT_Y410:
		case DXGI_FORMAT_YUY2:
			return 32;

		case DXGI_FORMAT_P010:
		case DXGI_FORMAT_P016:
			return 24;

		case DXGI_FORMAT_R8G8_TYPELESS:
		case DXGI_FORMAT_R8G8_UNORM:
		case DXGI_FORMAT_R8G8_UINT:
		case DXGI_FORMAT_R8G8_SNORM:
		case DXGI_FORMAT_R8G8_SINT:
		case DXGI_FORMAT_R16_TYPELESS:
		case DXGI_FORMAT_R16_FLOAT:
		case DXGI_FORMAT_D16_UNORM:
		case DXGI_FORMAT_R16_UNORM:
		case DXGI_FORMAT_R16_UINT:
		case DXGI_FORMAT_R16_SNORM:
		case DXGI_FORMAT_R16_SINT:
		case DXGI_FORMAT_B5G6R5_UNORM:
		case DXGI_FORMAT_B5G5R5A1_UNORM:
		case DXGI_FORMAT_A8P8:
		case DXGI_FORMAT_B4G4R4A4_UNORM:
			return 16;

		case DXGI_FORMAT_NV12:
		case DXGI_FORMAT_420_OPAQUE:
		case DXGI_FORMAT_NV11:
			return 12;

		case DXGI_FORMAT_R8_TYPELESS:
		case DXGI_FORMAT_R8_UNORM:
		case DXGI_FORMAT_R8_UINT:
		case DXGI_FORMAT_R8_SNORM:
		case DXGI_FORMAT_R8_SINT:
		case DXGI_FORMAT_A8_UNORM:
		case DXGI_FORMAT_BC2_TYPELESS:
		case DXGI_FORMAT_BC2_UNORM:
		case DXGI_FORMAT_BC2_UNORM_SRGB:
		case DXGI_FORMAT_BC3_TYPELESS:
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC3_UNORM_SRGB:
		case DXGI_FORMAT_BC5_TYPELESS:
		case DXGI_FORMAT_BC5_UNORM:
		case DXGI_FORMAT_BC5_SNORM:
		case DXGI_FORMAT_BC6H_TYPELESS:
		case DXGI_FORMAT_BC6H_UF16:
		case DXGI_FORMAT_BC6H_SF16:
		case DXGI_FORMAT_BC7_TYPELESS:
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
		case DXGI_FORMAT_AI44:
		case DXGI_FORMAT_IA44:
		case DXGI_FORMAT_P8:
			return 8;

		case DXGI_FORMAT_R1_UNORM:
			return 1;

		case DXGI_FORMAT_BC1_TYPELESS:
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC1_UNORM_SRGB:
		case DXGI_FORMAT_BC4_TYPELESS:
		case DXGI_FORMAT_BC4_UNORM:
		case DXGI_FORMAT_BC4_SNORM:
			return 4;

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

	bool IsDepth(DXGI_FORMAT format)
	{
		switch (format)
		{
		case DXGI_FORMAT_D32_FLOAT:
		case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
		case DXGI_FORMAT_D16_UNORM:
			return true;

		default:
			return false;
		}
	}

	uint32_t GetArraySize(const TextureDimension& dimension, const uint32_t depth)
	{
		if (dimension == TextureDimension::Texture2D)
		{
			return 1;
		}
		else if (dimension == TextureDimension::TextureCube)
		{
			return 6;
		}
		return std::min(1u, depth);
	}

	bool GfxTextureDX11::Initialize(const TextureProperties& properties)
	{
		m_Index = ++s_MaxIndex;
		m_Format = static_cast<DXGI_FORMAT>(properties.format);
		m_Width = properties.width;
		m_Height = properties.height;
		m_Depth = properties.depth;

		if (properties.data != nullptr)
		{
			uint32_t bitsPerPixel = GetBitsPerPixel(m_Format);
			uint32_t arraySize = GetArraySize(properties.dimension, properties.depth);
			uint32_t mipLevels = std::max(1u, properties.mipCount);
			uint32_t size = arraySize * mipLevels;
			D3D11_SUBRESOURCE_DATA* subresourceDatas = BB_MALLOC_ARRAY(D3D11_SUBRESOURCE_DATA, size);

			uint8_t* ptr = static_cast<uint8_t*>(properties.data);
			if (IsCompressed(m_Format))
			{
				uint32_t blockSize = bitsPerPixel * 16 / 8;
				for (uint32_t i = 0; i < arraySize; ++i)
				{
					uint32_t width = properties.width;
					uint32_t height = properties.height;
					for (uint32_t j = 0; j < mipLevels; ++j)
					{
						D3D11_SUBRESOURCE_DATA subresourceData;
						subresourceData.pSysMem = ptr;
						subresourceData.SysMemPitch = std::max(1u, (width + 3) / 4) * blockSize;
						subresourceData.SysMemSlicePitch = 0;
						subresourceDatas[D3D11CalcSubresource(j, i, mipLevels)] = subresourceData;
						ptr += subresourceData.SysMemPitch * std::max(1u, (height + 3) / 4);
						width /= 2;
						height /= 2;
					}
				}
			}
			else
			{
				for (uint32_t i = 0; i < arraySize; ++i)
				{
					uint32_t width = properties.width;
					uint32_t height = properties.height;
					for (uint32_t j = 0; j < mipLevels; ++j)
					{
						D3D11_SUBRESOURCE_DATA subresourceData;
						subresourceData.pSysMem = ptr;
						subresourceData.SysMemPitch = width * bitsPerPixel / 8;
						subresourceData.SysMemSlicePitch = properties.depth > 0 ? (subresourceData.SysMemPitch * height) : 0;
						subresourceDatas[D3D11CalcSubresource(j, i, mipLevels)] = subresourceData;
						ptr += width * height * bitsPerPixel / 8;
						width /= 2;
						height /= 2;
					}
				}
			}
			bool result = Initialize(subresourceDatas, size, properties);
			BB_FREE(subresourceDatas);
			return result;
		}
		else
		{
			return Initialize(nullptr, 0, properties);
		}
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

	ID3D11RenderTargetView* GfxTextureDX11::GetRTV(const uint32_t& slice)
	{
		return m_SlicesRenderTargetViews[slice].Get();
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

	void GfxTextureDX11::GetData(void* target, const Rectangle& area)
	{
		if (m_StagingTexture.Get() == nullptr || IsCompressed(m_Format))
		{
			BB_ERROR("The texture cannot be readed.");
			return;
		}
		m_DeviceContext->CopyResource(m_StagingTexture.Get(), m_Texture.Get());

		D3D11_MAPPED_SUBRESOURCE mappedTexture;
		ZeroMemory(&mappedTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));
		HRESULT hr = m_DeviceContext->Map(m_StagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedTexture);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get texture data."));
			return;
		}
		for (int i = 0; i < area.height; i++)
		{
			size_t pixelSize = mappedTexture.RowPitch / m_Width;
			size_t offset = ((area.y + i) * m_Width + area.x) * pixelSize;
			char* copyPtr = static_cast<char*>(mappedTexture.pData) + offset;
			char* targetPtr = static_cast<char*>(target) + (area.width * pixelSize * i);
			memcpy(targetPtr, copyPtr, area.width * pixelSize);
		}
		m_DeviceContext->Unmap(m_StagingTexture.Get(), 0);
	}

	void GfxTextureDX11::GetData(void* target)
	{
		if (m_StagingTexture.Get() == nullptr || IsCompressed(m_Format))
		{
			BB_ERROR("The texture cannot be readed.");
			return;
		}
		m_DeviceContext->CopyResource(m_StagingTexture.Get(), m_Texture.Get());

		uint32_t bytesPerPixel = GetBitsPerPixel(m_Format) / 8;
		uint8_t* ptr = static_cast<uint8_t*>(target);
		D3D11_MAPPED_SUBRESOURCE mappedTexture;
		ZeroMemory(&mappedTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));

		for (uint32_t i = 0; i < m_ArraySize; ++i)
		{
			uint32_t width = m_Width;
			uint32_t height = m_Height;
			for (uint32_t j = 0; j < m_MipLevels; ++j)
			{
				UINT subresource = D3D11CalcSubresource(j, i, m_MipLevels);
				HRESULT hr = m_DeviceContext->Map(m_StagingTexture.Get(), subresource, D3D11_MAP_READ, 0, &mappedTexture);
				if (FAILED(hr))
				{
					BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get texture data."));
					return;
				}
				uint8_t* src = static_cast<uint8_t*>(mappedTexture.pData);
				uint32_t dataSize = width * bytesPerPixel;
				for (uint32_t k = 0; k < height; ++k)
				{
					memcpy(ptr, src, dataSize);
					src += mappedTexture.RowPitch;
					ptr += dataSize;
				}
				m_DeviceContext->Unmap(m_StagingTexture.Get(), subresource);
				width /= 2;
				height /= 2;
			}
		}
	}

	void GfxTextureDX11::SetData(void* data, const size_t& size)
	{
		if (m_StagingTexture.Get() == nullptr)
		{
			BB_ERROR("The texture cannot be writed.");
			return;
		}
		D3D11_MAPPED_SUBRESOURCE mappedBuffer;
		ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

		m_DeviceContext->Map(m_StagingTexture.Get(), 0, D3D11_MAP_WRITE, 0, &mappedBuffer);
		memcpy(mappedBuffer.pData, data, size);
		m_DeviceContext->Unmap(m_StagingTexture.Get(), 0);

		m_DeviceContext->CopyResource(m_Texture.Get(), m_StagingTexture.Get());
	}

	void GfxTextureDX11::GenerateMipMaps()
	{
		m_DeviceContext->GenerateMips(m_ResourceView.Get());
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
		switch (filterMode)
		{
		case FilterMode::Point:
			return D3D11_FILTER_MIN_MAG_MIP_POINT;
		case FilterMode::Bilinear:
			return D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
		case FilterMode::Trilinear:
			return D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		case FilterMode::Anisotropic:
			return D3D11_FILTER_ANISOTROPIC;
		case FilterMode::CompareDepth:
			return D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
		default:
			return D3D11_FILTER_MIN_MAG_MIP_POINT;
		}
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

	DXGI_FORMAT GetTextureFormat(const DXGI_FORMAT& format)
	{
		if (format == DXGI_FORMAT_D24_UNORM_S8_UINT)
		{
			return DXGI_FORMAT_R24G8_TYPELESS;
		}
		else if (format == DXGI_FORMAT_D32_FLOAT)
		{
			return DXGI_FORMAT_R32_TYPELESS;
		}
		/*else if (format == DXGI_FORMAT_BC6H_UF16 || format == DXGI_FORMAT_BC6H_SF16)
		{
			return DXGI_FORMAT_BC6H_TYPELESS;
		}*/
		return format;
	}

	DXGI_FORMAT GetSRVFormat(const DXGI_FORMAT& format)
	{
		if (format == DXGI_FORMAT_D24_UNORM_S8_UINT)
		{
			return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
		}
		else if (format == DXGI_FORMAT_D32_FLOAT)
		{
			return DXGI_FORMAT_R32_FLOAT;
		}
		return format;
	}

	bool GfxTextureDX11::Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, const uint32_t& subresourceCount, const TextureProperties& properties)
	{
		uint32_t antiAliasing = std::max(1u, properties.antiAliasing);
		uint32_t arraySize = GetArraySize(properties.dimension, properties.depth);
		uint32_t mipLevels = std::max(1u, properties.mipCount);
		bool isCubemap = properties.dimension == TextureDimension::TextureCube;
		bool isArray = !isCubemap && arraySize > 1;
		bool isVolume = properties.dimension == TextureDimension::Texture3D;
		
		bool useDSV = IsDepth(m_Format);
		bool useRTV = !useDSV && properties.isRenderTarget;
		bool useUAV = properties.isUnorderedAccess;
		bool useStaging = properties.isReadable || properties.isWritable;
		bool isResource = !useDSV && !useRTV && !useUAV && !useStaging;
		bool useSampler = isResource || antiAliasing <= 1;

		m_ArraySize = std::max(1u, arraySize);
		m_MipLevels = mipLevels;

		// Texture
		if (isVolume)
		{
			D3D11_TEXTURE3D_DESC textureDesc;
			ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE3D_DESC));

			textureDesc.Width = m_Width;
			textureDesc.Height = m_Height;
			textureDesc.Depth = m_Depth;
			textureDesc.Usage = isResource ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT;
			textureDesc.CPUAccessFlags = 0;
			textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			textureDesc.MipLevels = 1;
			textureDesc.Format = GetTextureFormat(m_Format);
			textureDesc.MiscFlags = 0;

			if (useRTV)
			{
				textureDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
			}
			if (useUAV)
			{
				textureDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
			}

			HRESULT hr = m_Device->CreateTexture3D(&textureDesc, subresourceData, reinterpret_cast<ID3D11Texture3D**>(m_Texture.GetAddressOf()));
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
				return false;
			}
		}
		else
		{
			D3D11_TEXTURE2D_DESC textureDesc;
			ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

			textureDesc.Width = m_Width;
			textureDesc.Height = m_Height;
			textureDesc.Usage = isResource ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT;
			textureDesc.CPUAccessFlags = 0;
			textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			textureDesc.MipLevels = mipLevels;
			textureDesc.Format = GetTextureFormat(m_Format);
			textureDesc.SampleDesc.Count = antiAliasing;
			textureDesc.SampleDesc.Quality = GetQualityLevel(m_Format, antiAliasing);
			textureDesc.ArraySize = arraySize;
			textureDesc.MiscFlags = 0;
			
			if (useRTV)
			{
				textureDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
				m_SlicesRenderTargetViews.resize(arraySize);
				if (properties.generateMipMaps)
				{
					textureDesc.MipLevels = 0;
					textureDesc.MiscFlags |= D3D11_RESOURCE_MISC_GENERATE_MIPS;
				}
			}
			if (useDSV)
			{
				textureDesc.BindFlags |= D3D11_BIND_DEPTH_STENCIL;
			}
			if (useUAV)
			{
				textureDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
			}
			if (isCubemap)
			{
				textureDesc.MiscFlags |= D3D11_RESOURCE_MISC_TEXTURECUBE;
			}

			HRESULT hr = m_Device->CreateTexture2D(&textureDesc, subresourceData, reinterpret_cast<ID3D11Texture2D**>(m_Texture.GetAddressOf()));
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
				return false;
			}
		}

		// SRV
		D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
		ZeroMemory(&resourceViewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));

		resourceViewDesc.Format = GetSRVFormat(m_Format);

		if (isVolume)
		{
			resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
			resourceViewDesc.Texture3D.MipLevels = mipLevels;
			resourceViewDesc.Texture3D.MostDetailedMip = 0;
		}
		else if (isArray)
		{
			if (antiAliasing > 1)
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
				resourceViewDesc.Texture2DMSArray.ArraySize = arraySize;
			}
			else
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
				resourceViewDesc.Texture2DArray.ArraySize = arraySize;
				resourceViewDesc.Texture2DArray.MipLevels = mipLevels;
				resourceViewDesc.Texture2DArray.FirstArraySlice = 0;
				resourceViewDesc.Texture2DArray.MostDetailedMip = 0;
			}
		}
		else if (isCubemap)
		{
			resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
			resourceViewDesc.TextureCube.MipLevels = mipLevels;
			resourceViewDesc.TextureCube.MostDetailedMip = 0;
		}
		else
		{
			if (antiAliasing > 1)
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
			}
			else
			{
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				resourceViewDesc.Texture2D.MostDetailedMip = 0;
				resourceViewDesc.Texture2D.MipLevels = mipLevels;
			}
		}

		HRESULT hr = m_Device->CreateShaderResourceView(m_Texture.Get(), &resourceViewDesc, m_ResourceView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
			return false;
		}

		// RTV
		if (useRTV)
		{
			D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
			ZeroMemory(&renderTargetViewDesc, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));

			renderTargetViewDesc.Format = m_Format;
			if (isVolume)
			{
				uint32_t depth = properties.depth;
				m_SlicesRenderTargetViews.resize(depth);
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
				renderTargetViewDesc.Texture3D.MipSlice = 0;
				renderTargetViewDesc.Texture3D.FirstWSlice = 0;
				renderTargetViewDesc.Texture3D.WSize = -1;

				D3D11_RENDER_TARGET_VIEW_DESC sliceRenderTargetViewDesc = renderTargetViewDesc;
				for (uint32_t i = 0; i < depth; ++i)
				{
					sliceRenderTargetViewDesc.Texture3D.MipSlice = 0;
					sliceRenderTargetViewDesc.Texture3D.FirstWSlice = i;
					sliceRenderTargetViewDesc.Texture3D.WSize = 1;

					hr = m_Device->CreateRenderTargetView(m_Texture.Get(), &sliceRenderTargetViewDesc, m_SlicesRenderTargetViews[i].GetAddressOf());
					if (FAILED(hr))
					{
						BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
						return false;
					}
				}
			}
			else if (isArray)
			{
				m_SlicesRenderTargetViews.resize(arraySize);
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
			else if (isCubemap)
			{
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
				renderTargetViewDesc.Texture2DArray.MipSlice = 0;
				renderTargetViewDesc.Texture2DArray.FirstArraySlice = 0;
				renderTargetViewDesc.Texture2DArray.ArraySize = 6;

				m_SlicesRenderTargetViews.resize(mipLevels);

				D3D11_RENDER_TARGET_VIEW_DESC sliceRenderTargetViewDesc = renderTargetViewDesc;
				for (uint32_t i = 0; i < mipLevels; ++i)
				{
					sliceRenderTargetViewDesc.Texture2DArray.MipSlice = i;
					sliceRenderTargetViewDesc.Texture2DArray.FirstArraySlice = 0;
					sliceRenderTargetViewDesc.Texture2DArray.ArraySize = 6;

					hr = m_Device->CreateRenderTargetView(m_Texture.Get(), &sliceRenderTargetViewDesc, m_SlicesRenderTargetViews[i].GetAddressOf());
					if (FAILED(hr))
					{
						BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
						return false;
					}
				}
			}
			else
			{
				renderTargetViewDesc.ViewDimension = antiAliasing > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
				renderTargetViewDesc.Texture2D.MipSlice = 0;
			}

			hr = m_Device->CreateRenderTargetView(m_Texture.Get(), &renderTargetViewDesc, m_RenderTargetView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
				return false;
			}
		}

		// DSV
		if (useDSV)
		{
			D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
			ZeroMemory(&depthStencilViewDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

			depthStencilViewDesc.Format = m_Format;

			if (isArray)
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
			else
			{
				depthStencilViewDesc.ViewDimension = antiAliasing > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
				depthStencilViewDesc.Texture2D.MipSlice = 0;
			}

			hr = m_Device->CreateDepthStencilView(m_Texture.Get(), &depthStencilViewDesc, m_DepthStencilView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil view."));
				return false;
			}
		}

		// UAV
		if (useUAV)
		{
			D3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc;
			ZeroMemory(&unorderedAccessViewDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));

			unorderedAccessViewDesc.Format = static_cast<DXGI_FORMAT>(properties.format);

			if (isVolume)
			{
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
				unorderedAccessViewDesc.Texture3D.MipSlice = 0;
				unorderedAccessViewDesc.Texture3D.FirstWSlice = 0;
				unorderedAccessViewDesc.Texture3D.WSize = -1;
			}
			else if (isArray)
			{
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
				unorderedAccessViewDesc.Texture2DArray.MipSlice = 0;
				unorderedAccessViewDesc.Texture2DArray.FirstArraySlice = 0;
				unorderedAccessViewDesc.Texture2DArray.ArraySize = arraySize;
			}
			else
			{
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
				unorderedAccessViewDesc.Texture2D.MipSlice = 0;
			}

			hr = m_Device->CreateUnorderedAccessView(m_Texture.Get(), &unorderedAccessViewDesc, &m_UnorderedAccessView);
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create unordered access view."));
				return false;
			}
		}

		// Staging
		if (useStaging)
		{
			D3D11_TEXTURE2D_DESC textureDesc;
			ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

			textureDesc.Width = m_Width;
			textureDesc.Height = m_Height;
			textureDesc.MipLevels = mipLevels;
			textureDesc.ArraySize = arraySize;
			textureDesc.Format = m_Format;
			textureDesc.SampleDesc.Count = 1;
			textureDesc.MiscFlags = 0;

			textureDesc.Usage = D3D11_USAGE_STAGING;
			textureDesc.BindFlags = 0;
			textureDesc.CPUAccessFlags = (properties.isReadable ? D3D11_CPU_ACCESS_READ : 0) | (properties.isWritable ? D3D11_CPU_ACCESS_WRITE : 0);

			if (isCubemap)
			{
				textureDesc.MiscFlags |= D3D11_RESOURCE_MISC_TEXTURECUBE;
			}

			HRESULT hr = m_Device->CreateTexture2D(&textureDesc, nullptr, reinterpret_cast<ID3D11Texture2D**>(m_StagingTexture.GetAddressOf()));
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
				return false;
			}
		}

		// Sampler
		if (useSampler)
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
			samplerDesc.MaxAnisotropy = filter == D3D11_FILTER_ANISOTROPIC ? 8 : 1;
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
		
		return true;
	}
}