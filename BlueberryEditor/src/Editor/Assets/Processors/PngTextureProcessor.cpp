#include "PngTextureProcessor.h"

#include "Blueberry\Tools\StringConverter.h"
#include "BCFlipper.h"

namespace Blueberry
{
	ComPtr<ID3D11Device> PngTextureProcessor::s_Device = nullptr;

	PngTextureProcessor::~PngTextureProcessor()
	{
		if (m_ScratchImage.GetPixelsSize() > 0)
		{
			m_ScratchImage.Release();
		}
	}

	void PngTextureProcessor::Load(const String& path, const bool& srgb, const bool& generateMips)
	{
		HRESULT hr = DirectX::LoadFromWICFile(StringConverter::StringToWide(path).c_str(), (srgb ? DirectX::WIC_FLAGS_DEFAULT_SRGB : DirectX::WIC_FLAGS_IGNORE_SRGB) | DirectX::WIC_FLAGS_FORCE_RGB, nullptr, m_ScratchImage);
		if (FAILED(hr))
		{
			BB_ERROR("Failed to load texture from file.");
			return;
		}
		DirectX::ScratchImage flippedScratchImage;
		hr = DirectX::FlipRotate(m_ScratchImage.GetImages(), m_ScratchImage.GetImageCount(), m_ScratchImage.GetMetadata(), DirectX::TEX_FR_FLIP_VERTICAL, flippedScratchImage);
		if (FAILED(hr))
		{
			BB_ERROR("Failed to flip texture.");
			return;
		}
		m_ScratchImage = std::move(flippedScratchImage);

		DirectX::Image image = *m_ScratchImage.GetImages();
		m_Properties = {};
		m_Properties.width = static_cast<uint32_t>(image.width);
		m_Properties.height = static_cast<uint32_t>(image.height);
		m_Properties.mipCount = GetMipCount(static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height), generateMips);
	}

	void FlipDDS(const DXGI_FORMAT& format, unsigned char* image, const uint32_t& mipCount, uint32_t width, uint32_t height)
	{
		uint32_t blockSize = 0;
		switch (format)
		{
		case DXGI_FORMAT_BC1_UNORM:
			blockSize = 8;
			break;
		case DXGI_FORMAT_BC3_UNORM:
			blockSize = 16;
			break;
		case DXGI_FORMAT_BC4_UNORM:
			blockSize = 8;
			break;
		case DXGI_FORMAT_BC5_UNORM:
			blockSize = 16;
			break;
		}

		for (uint32_t i = 0; i < mipCount; ++i)
		{
			switch (format)
			{
			case DXGI_FORMAT_BC1_UNORM:
				FlipBC1Image(image, width, height);
				break;
			case DXGI_FORMAT_BC3_UNORM:
				FlipBC3Image(image, width, height);
				break;
			case DXGI_FORMAT_BC4_UNORM:
				FlipBC4Image(image, width, height);
				break;
			case DXGI_FORMAT_BC5_UNORM:
				FlipBC5Image(image, width, height);
				break;
			default:
				BB_ERROR("Can't flip image");
				return;
			}
			image += ((width + 3) / 4) * ((height + 3) / 4) * blockSize;
			width /= 2;
			height /= 2;
		}
	}

	void PngTextureProcessor::LoadDDS(const String& path)
	{
		HRESULT hr = DirectX::LoadFromDDSFile(StringConverter::StringToWide(path).c_str(), DirectX::DDS_FLAGS::DDS_FLAGS_NONE, nullptr, m_ScratchImage);
		if (FAILED(hr))
		{
			BB_ERROR("Failed to load texture from file.");
			return;
		}

		DirectX::Image image = *m_ScratchImage.GetImages();
		FlipDDS(m_ScratchImage.GetMetadata().format, m_ScratchImage.GetPixels(), static_cast<uint32_t>(m_ScratchImage.GetImageCount()), static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height));

		m_Properties = {};
		m_Properties.width = static_cast<uint32_t>(image.width);
		m_Properties.height = static_cast<uint32_t>(image.height);
		m_Properties.mipCount = static_cast<uint32_t>(m_ScratchImage.GetImageCount());
	}

	void PngTextureProcessor::LoadHDR(const String& path)
	{
		HRESULT hr = DirectX::LoadFromHDRFile(StringConverter::StringToWide(path).c_str(), nullptr, m_ScratchImage);
		if (FAILED(hr))
		{
			BB_ERROR("Failed to load texture from file.");
			return;
		}

		DirectX::ScratchImage flippedScratchImage;
		hr = DirectX::FlipRotate(m_ScratchImage.GetImages(), m_ScratchImage.GetImageCount(), m_ScratchImage.GetMetadata(), DirectX::TEX_FR_FLIP_VERTICAL, flippedScratchImage);
		if (FAILED(hr))
		{
			BB_ERROR("Failed to flip texture.");
			return;
		}
		m_ScratchImage = std::move(flippedScratchImage);

		DirectX::Image image = *m_ScratchImage.GetImages();
		m_Properties = {};
		m_Properties.width = static_cast<uint32_t>(image.width);
		m_Properties.height = static_cast<uint32_t>(image.height);
		m_Properties.mipCount = static_cast<uint32_t>(m_ScratchImage.GetImageCount());
	}

	void PngTextureProcessor::CreateCube(const TextureFormat& format, const uint32_t& width, const uint32_t& height)
	{
		DXGI_FORMAT dxgiFormat = static_cast<DXGI_FORMAT>(format);
		m_ScratchImage.InitializeCube(dxgiFormat, width, height, 1, 1);

		m_Properties = {};
		m_Properties.width = width;
		m_Properties.height = height;
		m_Properties.mipCount = 1;
	}

	void PngTextureProcessor::Compress(const TextureFormat& format)
	{
		DXGI_FORMAT dxgiFormat = static_cast<DXGI_FORMAT>(format);
		const DirectX::Image* image = m_ScratchImage.GetImage(0, 0, 0);
		if (DirectX::IsSRGB(image->format))
		{
			dxgiFormat = DirectX::MakeSRGB(dxgiFormat);
		}
		if (!DirectX::IsCompressed(dxgiFormat) || (image->width % 4 > 0) || (image->height % 4 > 0))
		{
			return;
		}
		if (m_Properties.mipCount > 1)
		{
			DirectX::ScratchImage mipmappedScratchImage;
			HRESULT hr = DirectX::GenerateMipMaps(*m_ScratchImage.GetImages(), DirectX::TEX_FILTER_FANT, m_Properties.mipCount, mipmappedScratchImage);
			m_ScratchImage = std::move(mipmappedScratchImage);
		}

		if (s_Device == nullptr)
		{
			D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_DEBUG, NULL, 0, D3D11_SDK_VERSION, s_Device.GetAddressOf(), NULL, NULL);
		}

		DirectX::ScratchImage compressedScratchImage;
		HRESULT hr = DirectX::Compress(s_Device.Get(), m_ScratchImage.GetImages(), m_ScratchImage.GetImageCount(), m_ScratchImage.GetMetadata(), dxgiFormat, DirectX::TEX_COMPRESS_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, compressedScratchImage);
		if (FAILED(hr))
		{
			BB_ERROR("Failed to compress texture.");
			return;
		}
		m_ScratchImage = std::move(compressedScratchImage);
	}

	const PngTextureProperties& PngTextureProcessor::GetProperties()
	{
		DirectX::Image image = *m_ScratchImage.GetImages();
		m_Properties.data = m_ScratchImage.GetPixels();
		m_Properties.dataSize = m_ScratchImage.GetPixelsSize();
		m_Properties.format = static_cast<TextureFormat>(image.format);
		return m_Properties;
	}

	uint8_t* PngTextureProcessor::GetData()
	{
		return m_ScratchImage.GetImages()->pixels;
	}

	const size_t PngTextureProcessor::GetDataSize()
	{
		return m_ScratchImage.GetPixelsSize();
	}
}
