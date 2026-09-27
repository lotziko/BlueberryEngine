#include "GfxTextureDX11.h"

#include "..\Windows\WindowsHelper.h"
#include "..\Windows\DxgiHelper.h"

namespace Blueberry
{
	GfxPointerCache<GfxTextureDX11> GfxTextureDX11::s_PointerCache = {};

	GfxTextureDX11::GfxTextureDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
		m_Index = s_PointerCache.Allocate(this);
	}

	GfxTextureDX11::~GfxTextureDX11()
	{
		s_PointerCache.Deallocate(m_Index);
		m_SlicesRenderTargetViews.clear();
	}

	bool GfxTextureDX11::Initialize(const TextureProperties& properties)
	{
		m_Format = properties.format;
		m_DxgiFormat = static_cast<DXGI_FORMAT>(properties.format);
		m_Dimension = properties.dimension;
		m_Width = properties.width;
		m_Height = properties.height;
		m_Depth = properties.depth;
		m_FilterMode = properties.filterMode;
		m_WrapMode = properties.wrapMode;
		m_AntiAliasing = std::max(1u, properties.antiAliasing);
		m_ArraySize = DxgiHelper::GetArraySize(properties.dimension, properties.depth);
		m_MipLevels = std::max(1u, properties.mipCount);

		if (properties.data != nullptr)
		{
			uint32_t bitsPerPixel = DxgiHelper::GetBitsPerPixel(m_DxgiFormat);
			uint32_t arraySize = DxgiHelper::GetArraySize(properties.dimension, properties.depth);
			uint32_t mipLevels = std::max(1u, properties.mipCount);
			uint32_t size = arraySize * mipLevels;
			List<D3D11_SUBRESOURCE_DATA> subresourceDatas(size);

			const uint8_t* ptr = static_cast<const uint8_t*>(properties.data);
			if (DxgiHelper::IsCompressed(m_DxgiFormat))
			{
				uint32_t blockSize = bitsPerPixel * 16 / 8;
				for (uint32_t i = 0; i < arraySize; ++i)
				{
					uint32_t width = properties.width;
					uint32_t height = properties.height;
					for (uint32_t j = 0; j < mipLevels; ++j)
					{
						D3D11_SUBRESOURCE_DATA subresourceData = {};
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
						D3D11_SUBRESOURCE_DATA subresourceData = {};
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
			bool result = Initialize(subresourceDatas.data(), size, properties);
			return result;
		}
		else
		{
			return Initialize(nullptr, 0, properties);
		}
	}

	ID3D11Resource* GfxTextureDX11::GetResource() const
	{
		return m_Resource.Get();
	}

	ID3D11ShaderResourceView* GfxTextureDX11::GetShaderResourceView() const
	{
		return m_ShaderResourceView.Get();
	}

	ID3D11ShaderResourceView* GfxTextureDX11::GetShaderResourceView(uint32_t arraySlice, uint32_t mipSlice)
	{
		if (m_SlicesShaderResourceViews.size() == 0)
		{
			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				m_SlicesShaderResourceViews.resize(m_MipLevels);
				break;
			case TextureDimension::Texture2DArray:
				m_SlicesShaderResourceViews.resize(static_cast<size_t>(m_ArraySize * m_MipLevels));
				break;
			case TextureDimension::TextureCube:
				m_SlicesShaderResourceViews.resize(static_cast<size_t>(6 * m_MipLevels));
				break;
			case TextureDimension::Texture3D:
				m_SlicesShaderResourceViews.resize(static_cast<size_t>(m_Depth * m_MipLevels));
				break;
			}
		}

		uint32_t index = arraySlice * m_MipLevels + mipSlice;
		ID3D11ShaderResourceView* shaderResourceView = m_SlicesShaderResourceViews[index].Get();
		if (shaderResourceView == nullptr)
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc = {};
			shaderResourceViewDesc.Format = m_DxgiFormat;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				if (m_AntiAliasing > 1)
				{
					shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
				}
				else
				{
					shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
					shaderResourceViewDesc.Texture2D.MostDetailedMip = mipSlice;
					shaderResourceViewDesc.Texture2D.MipLevels = 1;
				}
				break;
			case TextureDimension::Texture2DArray:
				if (m_AntiAliasing > 1)
				{
					shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
					shaderResourceViewDesc.Texture2DMSArray.FirstArraySlice = arraySlice;
					shaderResourceViewDesc.Texture2DMSArray.ArraySize = 1;
				}
				else
				{
					shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
					shaderResourceViewDesc.Texture2DArray.MostDetailedMip = mipSlice;
					shaderResourceViewDesc.Texture2DArray.MipLevels = 1;
					shaderResourceViewDesc.Texture2DArray.FirstArraySlice = arraySlice;
					shaderResourceViewDesc.Texture2DArray.ArraySize = 1;
				}
				break;
			case TextureDimension::TextureCube:
				shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
				shaderResourceViewDesc.TextureCube.MostDetailedMip = mipSlice;
				shaderResourceViewDesc.TextureCube.MipLevels = 1;
				break;
			case TextureDimension::Texture3D:
				shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
				shaderResourceViewDesc.Texture3D.MostDetailedMip = mipSlice;
				shaderResourceViewDesc.Texture3D.MipLevels = 1;
				break;
			}

			HRESULT hr = m_Device->CreateShaderResourceView(m_Resource.Get(), &shaderResourceViewDesc, m_SlicesShaderResourceViews[index].GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
				return false;
			}
			return m_SlicesShaderResourceViews[index].Get();
		}
		return shaderResourceView;
	}

	ID3D11RenderTargetView* GfxTextureDX11::GetRenderTargetView() const
	{
		return m_RenderTargetView.Get();
	}

	ID3D11RenderTargetView* GfxTextureDX11::GetRenderTargetView(uint32_t arraySlice, uint32_t mipSlice)
	{
		if (m_SlicesRenderTargetViews.size() == 0)
		{
			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				m_SlicesRenderTargetViews.resize(m_MipLevels);
				break;
			case TextureDimension::Texture2DArray:
				m_SlicesRenderTargetViews.resize(static_cast<size_t>(m_ArraySize * m_MipLevels));
				break;
			case TextureDimension::TextureCube:
				m_SlicesRenderTargetViews.resize(static_cast<size_t>(6 * m_MipLevels));
				break;
			case TextureDimension::Texture3D:
				m_SlicesRenderTargetViews.resize(static_cast<size_t>(m_Depth * m_MipLevels));
				break;
			}
		}

		uint32_t index = arraySlice * m_MipLevels + mipSlice;
		ID3D11RenderTargetView* renderTargetView = m_SlicesRenderTargetViews[index].Get();
		if (renderTargetView == nullptr)
		{
			D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc = {};
			renderTargetViewDesc.Format = m_DxgiFormat;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				if (m_AntiAliasing <= 1)
				{
					renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
					renderTargetViewDesc.Texture2D.MipSlice = mipSlice;
				}
				break;
			case TextureDimension::Texture2DArray:
				if (m_AntiAliasing > 1)
				{
					renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
					renderTargetViewDesc.Texture2DMSArray.FirstArraySlice = arraySlice;
					renderTargetViewDesc.Texture2DMSArray.ArraySize = 1;
				}
				else
				{
					renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
					renderTargetViewDesc.Texture2DArray.MipSlice = mipSlice;
					renderTargetViewDesc.Texture2DArray.FirstArraySlice = arraySlice;
					renderTargetViewDesc.Texture2DArray.ArraySize = 1;
				}
				break;
			case TextureDimension::TextureCube:
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
				renderTargetViewDesc.Texture2DArray.MipSlice = mipSlice;
				renderTargetViewDesc.Texture2DArray.FirstArraySlice = arraySlice;
				renderTargetViewDesc.Texture2DArray.ArraySize = 1;
				break;
			case TextureDimension::Texture3D:
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
				renderTargetViewDesc.Texture3D.MipSlice = mipSlice;
				renderTargetViewDesc.Texture3D.FirstWSlice = arraySlice;
				renderTargetViewDesc.Texture3D.WSize = 1;
				break;
			}

			HRESULT hr = m_Device->CreateRenderTargetView(m_Resource.Get(), &renderTargetViewDesc, m_SlicesRenderTargetViews[index].GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
				return false;
			}
			return m_SlicesRenderTargetViews[index].Get();
		}
		return renderTargetView;
	}

	ID3D11DepthStencilView* GfxTextureDX11::GetDepthStencilView() const
	{
		return m_DepthStencilView.Get();
	}

	ID3D11UnorderedAccessView* GfxTextureDX11::GetUnorderedAccessView() const
	{
		return m_UnorderedAccessView.Get();
	}

	ID3D11UnorderedAccessView* GfxTextureDX11::GetUnorderedAccessView(uint32_t arraySlice, uint32_t mipSlice)
	{
		if (m_SlicesUnorderedAccessViews.size() == 0)
		{
			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				m_SlicesUnorderedAccessViews.resize(m_MipLevels);
				break;
			case TextureDimension::Texture2DArray:
				m_SlicesUnorderedAccessViews.resize(static_cast<size_t>(m_ArraySize * m_MipLevels));
				break;
			case TextureDimension::Texture3D:
				m_SlicesUnorderedAccessViews.resize(static_cast<size_t>(m_Depth * m_MipLevels));
				break;
			}
		}

		uint32_t index = arraySlice * m_MipLevels + mipSlice;
		ID3D11UnorderedAccessView* unorderedAccessView = m_SlicesUnorderedAccessViews[index].Get();
		if (unorderedAccessView == nullptr)
		{
			D3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc = {};
			unorderedAccessViewDesc.Format = m_DxgiFormat;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
				unorderedAccessViewDesc.Texture2D.MipSlice = mipSlice;
				break;
			case TextureDimension::Texture2DArray:
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
				unorderedAccessViewDesc.Texture2DArray.MipSlice = mipSlice;
				unorderedAccessViewDesc.Texture2DArray.FirstArraySlice = arraySlice;
				unorderedAccessViewDesc.Texture2DArray.ArraySize = 1;
				break;
			case TextureDimension::Texture3D:
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
				unorderedAccessViewDesc.Texture3D.MipSlice = mipSlice;
				unorderedAccessViewDesc.Texture3D.FirstWSlice = arraySlice;
				unorderedAccessViewDesc.Texture3D.WSize = -1;
				break;
			}

			HRESULT hr = m_Device->CreateUnorderedAccessView(m_Resource.Get(), &unorderedAccessViewDesc, m_SlicesUnorderedAccessViews[index].GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
				return false;
			}
			return m_SlicesUnorderedAccessViews[index].Get();
		}
		return unorderedAccessView;
	}

	ID3D11SamplerState* GfxTextureDX11::GetSamplerState() const
	{
		return m_SamplerState.Get();
	}

	void GfxTextureDX11::SetSamplerState(ID3D11SamplerState* samplerState)
	{
		m_SamplerState = samplerState;
	}

	const DXGI_FORMAT GfxTextureDX11::GetDxgiFormat() const
	{
		return m_DxgiFormat;
	}

	void* GfxTextureDX11::GetHandle()
	{
		return m_ShaderResourceView.Get();
	}

	void GfxTextureDX11::GetData(void* data, const Rectangle& area)
	{
		if (m_StagingTexture.Get() == nullptr || DxgiHelper::IsCompressed(m_DxgiFormat))
		{
			BB_ERROR("The texture cannot be readed.");
			return;
		}
		m_DeviceContext->CopyResource(m_StagingTexture.Get(), m_Resource.Get());

		D3D11_MAPPED_SUBRESOURCE mappedTexture = {};
		HRESULT hr = m_DeviceContext->Map(m_StagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedTexture);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get texture data."));
			return;
		}
		for (int i = 0; i < area.height; i++)
		{
			size_t pixelSize = mappedTexture.RowPitch / m_Width;
			size_t offset = (static_cast<size_t>(area.y + i) * m_Width + area.x) * pixelSize;
			char* copyPtr = static_cast<char*>(mappedTexture.pData) + offset;
			char* targetPtr = static_cast<char*>(data) + (area.width * pixelSize * i);
			memcpy(targetPtr, copyPtr, area.width * pixelSize);
		}
		m_DeviceContext->Unmap(m_StagingTexture.Get(), 0);
	}

	void GfxTextureDX11::GetData(void* data)
	{
		if (m_StagingTexture.Get() == nullptr || DxgiHelper::IsCompressed(m_DxgiFormat))
		{
			BB_ERROR("The texture cannot be readed.");
			return;
		}
		m_DeviceContext->CopyResource(m_StagingTexture.Get(), m_Resource.Get());

		uint32_t bytesPerPixel = DxgiHelper::GetBitsPerPixel(m_DxgiFormat) / 8;
		uint8_t* ptr = static_cast<uint8_t*>(data);
		D3D11_MAPPED_SUBRESOURCE mappedTexture = {};

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

	void GfxTextureDX11::SetData(void* data, size_t size)
	{
		if (m_StagingTexture.Get() == nullptr)
		{
			BB_ERROR("The texture cannot be writed.");
			return;
		}
		D3D11_MAPPED_SUBRESOURCE mappedTexture = {};
		m_DeviceContext->Map(m_StagingTexture.Get(), 0, D3D11_MAP_WRITE, 0, &mappedTexture);
		memcpy(mappedTexture.pData, data, size);
		m_DeviceContext->Unmap(m_StagingTexture.Get(), 0);

		m_DeviceContext->CopyResource(m_Resource.Get(), m_StagingTexture.Get());
	}

	void GfxTextureDX11::SetWrapMode(WrapMode wrapMode)
	{
		if (m_WrapMode != wrapMode)
		{
			m_WrapMode = wrapMode;
			m_SamplerState.Reset();
		}
	}

	void GfxTextureDX11::SetFilterMode(FilterMode filterMode)
	{
		if (m_FilterMode != filterMode)
		{
			m_FilterMode = filterMode;
			m_SamplerState.Reset();
		}
	}

	void GfxTextureDX11::SetName(const String& name)
	{
		m_Resource->SetPrivateData(WKPDID_D3DDebugObjectName, static_cast<UINT>(name.size()), name.data());
	}

	GfxTextureDX11* GfxTextureDX11::Get(uint32_t index)
	{
		return s_PointerCache.Get(index);
	}

	DXGI_FORMAT GetTextureFormat(DXGI_FORMAT format)
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

	uint32_t GetQualityLevel(ID3D11Device* device, DXGI_FORMAT format, uint32_t antiAliasing)
	{
		if (antiAliasing > 1)
		{
			uint32_t qualityLevels;
			HRESULT hr = device->CheckMultisampleQualityLevels(format, antiAliasing, &qualityLevels);
			return qualityLevels - 1;
		}
		return 0;
	}

	bool GfxTextureDX11::Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, uint32_t subresourceCount, const TextureProperties& properties)
	{
		bool useDSV = DxgiHelper::IsDepth(m_DxgiFormat);
		bool useRTV = !useDSV && HasFlag(properties.usageFlags, TextureUsageFlags::RenderTarget);
		bool useUAV = HasFlag(properties.usageFlags, TextureUsageFlags::UnorderedAccess);
		bool isReadable = HasFlag(properties.usageFlags, TextureUsageFlags::CPUReadable);
		bool isWritable = HasFlag(properties.usageFlags, TextureUsageFlags::CPUWritable);
		bool useStaging = isReadable || isWritable;
		bool isResource = !useDSV && !useRTV && !useUAV && !useStaging;
		bool useSampler = isResource || m_AntiAliasing <= 1;

		// Texture
		switch (m_Dimension)
		{
		case TextureDimension::Texture2D:
		case TextureDimension::Texture2DArray:
		case TextureDimension::TextureCube:
		case TextureDimension::TextureCubeArray:
		{
			D3D11_TEXTURE2D_DESC textureDesc = {};
			textureDesc.Width = properties.width;
			textureDesc.Height = properties.height;
			textureDesc.Usage = isResource ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT;
			textureDesc.CPUAccessFlags = 0;
			textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			textureDesc.MipLevels = m_MipLevels;
			textureDesc.Format = GetTextureFormat(m_DxgiFormat);
			textureDesc.SampleDesc.Count = m_AntiAliasing;
			textureDesc.SampleDesc.Quality = GetQualityLevel(m_Device, m_DxgiFormat, m_AntiAliasing);
			textureDesc.ArraySize = m_ArraySize;
			textureDesc.MiscFlags = 0;

			if (useRTV)
			{
				textureDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
			}
			if (useDSV)
			{
				textureDesc.BindFlags |= D3D11_BIND_DEPTH_STENCIL;
			}
			if (useUAV)
			{
				textureDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
			}
			if (m_Dimension == TextureDimension::TextureCube || m_Dimension == TextureDimension::TextureCubeArray)
			{
				textureDesc.MiscFlags |= D3D11_RESOURCE_MISC_TEXTURECUBE;
			}

			HRESULT hr = m_Device->CreateTexture2D(&textureDesc, subresourceData, reinterpret_cast<ID3D11Texture2D**>(m_Resource.GetAddressOf()));
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
				return false;
			}
		}
		break;
		case TextureDimension::Texture3D:
		{
			D3D11_TEXTURE3D_DESC textureDesc = {};
			textureDesc.Width = properties.width;
			textureDesc.Height = properties.height;
			textureDesc.Depth = properties.depth;
			textureDesc.Usage = isResource ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT;
			textureDesc.CPUAccessFlags = 0;
			textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			textureDesc.MipLevels = 1;
			textureDesc.Format = GetTextureFormat(m_DxgiFormat);
			textureDesc.MiscFlags = 0;

			if (useRTV)
			{
				textureDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
			}
			if (useUAV)
			{
				textureDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
			}

			HRESULT hr = m_Device->CreateTexture3D(&textureDesc, subresourceData, reinterpret_cast<ID3D11Texture3D**>(m_Resource.GetAddressOf()));
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
				return false;
			}
		}
		break;
		}

		// SRV
		D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc = {};
		shaderResourceViewDesc.Format = DxgiHelper::GetSRVFormat(m_DxgiFormat);

		switch (m_Dimension)
		{
		case TextureDimension::Texture2D:
			if (m_AntiAliasing > 1)
			{
				shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
			}
			else
			{
				shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
				shaderResourceViewDesc.Texture2D.MipLevels = m_MipLevels;
			}
			break;
		case TextureDimension::Texture2DArray:
			if (m_AntiAliasing > 1)
			{
				shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
				shaderResourceViewDesc.Texture2DMSArray.FirstArraySlice = 0;
				shaderResourceViewDesc.Texture2DMSArray.ArraySize = m_ArraySize;
			}
			else
			{
				shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
				shaderResourceViewDesc.Texture2DArray.MostDetailedMip = 0;
				shaderResourceViewDesc.Texture2DArray.MipLevels = m_MipLevels;
				shaderResourceViewDesc.Texture2DArray.FirstArraySlice = 0;
				shaderResourceViewDesc.Texture2DArray.ArraySize = m_ArraySize;
			}
			break;
		case TextureDimension::TextureCube:
			shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
			shaderResourceViewDesc.TextureCube.MostDetailedMip = 0;
			shaderResourceViewDesc.TextureCube.MipLevels = m_MipLevels;
			break;
		case TextureDimension::TextureCubeArray:
			shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBEARRAY;
			shaderResourceViewDesc.TextureCubeArray.MostDetailedMip = 0;
			shaderResourceViewDesc.TextureCubeArray.MipLevels = m_MipLevels;
			shaderResourceViewDesc.TextureCubeArray.First2DArrayFace = 0;
			shaderResourceViewDesc.TextureCubeArray.NumCubes = m_ArraySize / 6;
			break;
		case TextureDimension::Texture3D:
			shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
			shaderResourceViewDesc.Texture3D.MostDetailedMip = 0;
			shaderResourceViewDesc.Texture3D.MipLevels = m_MipLevels;
			break;
		}
		
		HRESULT hr = m_Device->CreateShaderResourceView(m_Resource.Get(), &shaderResourceViewDesc, m_ShaderResourceView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
			return false;
		}

		// RTV
		if (useRTV)
		{
			D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc = {};
			renderTargetViewDesc.Format = m_DxgiFormat;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				renderTargetViewDesc.ViewDimension = m_AntiAliasing > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
				renderTargetViewDesc.Texture2D.MipSlice = 0;
				break;
			case TextureDimension::Texture2DArray:
				if (m_AntiAliasing > 1)
				{
					renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
					renderTargetViewDesc.Texture2DMSArray.FirstArraySlice = 0;
					renderTargetViewDesc.Texture2DMSArray.ArraySize = m_ArraySize;
				}
				else
				{
					renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
					renderTargetViewDesc.Texture2DArray.MipSlice = 0;
					renderTargetViewDesc.Texture2DArray.FirstArraySlice = 0;
					renderTargetViewDesc.Texture2DArray.ArraySize = m_ArraySize;
				}
				break;
			case TextureDimension::TextureCube:
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
				renderTargetViewDesc.Texture2DArray.MipSlice = 0;
				renderTargetViewDesc.Texture2DArray.FirstArraySlice = 0;
				renderTargetViewDesc.Texture2DArray.ArraySize = 6;
				break;
			case TextureDimension::Texture3D:
				renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
				renderTargetViewDesc.Texture3D.MipSlice = 0;
				renderTargetViewDesc.Texture3D.FirstWSlice = 0;
				renderTargetViewDesc.Texture3D.WSize = -1;
				break;
			}

			hr = m_Device->CreateRenderTargetView(m_Resource.Get(), &renderTargetViewDesc, m_RenderTargetView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
				return false;
			}
		}

		// DSV
		if (useDSV)
		{
			D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc = {};
			depthStencilViewDesc.Format = m_DxgiFormat;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				depthStencilViewDesc.ViewDimension = m_AntiAliasing > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
				depthStencilViewDesc.Texture2D.MipSlice = 0;
				break;
			case TextureDimension::Texture2DArray:
				if (m_AntiAliasing > 1)
				{
					depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
					depthStencilViewDesc.Texture2DMSArray.FirstArraySlice = 0;
					depthStencilViewDesc.Texture2DMSArray.ArraySize = m_ArraySize;
				}
				else
				{
					depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
					depthStencilViewDesc.Texture2DArray.MipSlice = 0;
					depthStencilViewDesc.Texture2DArray.FirstArraySlice = 0;
					depthStencilViewDesc.Texture2DArray.ArraySize = m_ArraySize;
				}
				break;
			}

			hr = m_Device->CreateDepthStencilView(m_Resource.Get(), &depthStencilViewDesc, m_DepthStencilView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil view."));
				return false;
			}
		}

		// UAV
		if (useUAV)
		{
			D3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc = {};
			unorderedAccessViewDesc.Format = m_DxgiFormat;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
				unorderedAccessViewDesc.Texture2D.MipSlice = 0;
				break;
			case TextureDimension::Texture2DArray:
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
				unorderedAccessViewDesc.Texture2DArray.MipSlice = 0;
				unorderedAccessViewDesc.Texture2DArray.FirstArraySlice = 0;
				unorderedAccessViewDesc.Texture2DArray.ArraySize = m_ArraySize;
				break;
			case TextureDimension::Texture3D:
				unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
				unorderedAccessViewDesc.Texture3D.MipSlice = 0;
				unorderedAccessViewDesc.Texture3D.FirstWSlice = 0;
				unorderedAccessViewDesc.Texture3D.WSize = -1;
				break;
			}

			hr = m_Device->CreateUnorderedAccessView(m_Resource.Get(), &unorderedAccessViewDesc, &m_UnorderedAccessView);
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create unordered access view."));
				return false;
			}
		}

		// Staging
		if (useStaging)
		{
			D3D11_TEXTURE2D_DESC textureDesc = {};
			textureDesc.Width = properties.width;
			textureDesc.Height = properties.height;
			textureDesc.MipLevels = m_MipLevels;
			textureDesc.ArraySize = m_ArraySize;
			textureDesc.Format = m_DxgiFormat;
			textureDesc.SampleDesc.Count = 1;
			textureDesc.MiscFlags = 0;

			textureDesc.Usage = D3D11_USAGE_STAGING;
			textureDesc.BindFlags = 0;
			textureDesc.CPUAccessFlags = (isReadable ? D3D11_CPU_ACCESS_READ : 0) | (isWritable ? D3D11_CPU_ACCESS_WRITE : 0);

			if (m_Dimension == TextureDimension::TextureCube)
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

		return true;
	}
}