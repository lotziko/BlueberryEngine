#include "TextureHelper.h"

#include "Blueberry\Tools\StringConverter.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Concrete\Windows\ComPtr.h"
#include "BCFlipper.h"

#include <directxtex\DirectXTex.h>

namespace Blueberry
{
	static ComPtr<ID3D11Device> s_Device = nullptr;
	static Material* s_EquirectangularToCubemapMaterial = nullptr;
	static Material* s_GenerateReflectionMaterial = nullptr;

	void FlipDDS(const DXGI_FORMAT& format, uint8_t* image, const uint32_t& mipCount, uint32_t width, uint32_t height)
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

	void TextureHelper::Load(DirectX::ScratchImage& scratchImage, const String& path, const String& extension, const bool& srgb)
	{
		if (extension == ".dds")
		{
			HRESULT hr = DirectX::LoadFromDDSFile(StringConverter::StringToWide(path).c_str(), DirectX::DDS_FLAGS::DDS_FLAGS_NONE, nullptr, scratchImage);
			if (FAILED(hr))
			{
				BB_ERROR("Failed to load DDS texture from file.");
				return;
			}
		}
		else if (extension == ".hdr")
		{
			HRESULT hr = DirectX::LoadFromHDRFile(StringConverter::StringToWide(path).c_str(), nullptr, scratchImage);
			if (FAILED(hr))
			{
				BB_ERROR("Failed to load HDR texture from file.");
				return;
			}
		}
		else
		{
			HRESULT hr = DirectX::LoadFromWICFile(StringConverter::StringToWide(path).c_str(), srgb ? DirectX::WIC_FLAGS_NONE : DirectX::WIC_FLAGS_IGNORE_SRGB, nullptr, scratchImage);
			if (FAILED(hr))
			{
				BB_ERROR("Failed to load texture from file.");
				return;
			}
		}
	}

	void TextureHelper::Flip(DirectX::ScratchImage& scratchImage)
	{
		DXGI_FORMAT format = scratchImage.GetMetadata().format;
		if (DirectX::IsCompressed(format))
		{
			// May not work for multiple slice textures
			FlipDDS(format, scratchImage.GetPixels(), static_cast<uint32_t>(scratchImage.GetImageCount()), static_cast<uint32_t>(scratchImage.GetMetadata().width), static_cast<uint32_t>(scratchImage.GetMetadata().height));
		}
		else
		{
			DirectX::ScratchImage flippedScratchImage;
			HRESULT hr = DirectX::FlipRotate(scratchImage.GetImages(), scratchImage.GetImageCount(), scratchImage.GetMetadata(), DirectX::TEX_FR_FLIP_VERTICAL, flippedScratchImage);
			if (FAILED(hr))
			{
				BB_ERROR("Failed to flip texture.");
				return;
			}
			scratchImage = std::move(flippedScratchImage);
		}
	}

	void TextureHelper::GenerateMipMaps(DirectX::ScratchImage& scratchImage)
	{
		DirectX::ScratchImage mipmappedScratchImage;
		HRESULT hr = DirectX::GenerateMipMaps(*scratchImage.GetImages(), DirectX::TEX_FILTER_FANT, GetMipCount(static_cast<uint32_t>(scratchImage.GetMetadata().width), static_cast<uint32_t>(scratchImage.GetMetadata().height), true), mipmappedScratchImage);
		if (mipmappedScratchImage.GetImageCount() > 0)
		{
			scratchImage = std::move(mipmappedScratchImage);
		}
	}

	bool CanCompressOnGPU(DXGI_FORMAT format)
	{
		switch (format)
		{
		case DXGI_FORMAT_BC6H_UF16:
		case DXGI_FORMAT_BC6H_SF16:
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
			return true;
		default:
			return false;
		}
	}

	void TextureHelper::Compress(DirectX::ScratchImage& scratchImage, const TextureFormat& format, const bool& srgb)
	{
		DXGI_FORMAT dxgiFormat = static_cast<DXGI_FORMAT>(format);
		if (scratchImage.GetMetadata().format == dxgiFormat)
		{
			return;
		}
		const DirectX::Image* image = scratchImage.GetImage(0, 0, 0);
		if (DirectX::IsCompressed(dxgiFormat) && (image->width % 4 > 0) && (image->height % 4 > 0))
		{
			DirectX::ScratchImage resizedScratchImage;
			DirectX::Resize(*image, image->width + (4 - image->width % 4), image->height + (4 - image->height % 4), DirectX::TEX_FILTER_FLAGS::TEX_FILTER_DEFAULT, resizedScratchImage);
			scratchImage = std::move(resizedScratchImage);
		}

		if (s_Device == nullptr)
		{
			D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_DEBUG, NULL, 0, D3D11_SDK_VERSION, s_Device.GetAddressOf(), NULL, NULL);
		}

		DirectX::ScratchImage compressedScratchImage;
		HRESULT hr;
		if (DirectX::IsCompressed(dxgiFormat))
		{
			if (CanCompressOnGPU(dxgiFormat))
			{
				hr = DirectX::Compress(s_Device.Get(), scratchImage.GetImages(), scratchImage.GetImageCount(), scratchImage.GetMetadata(), dxgiFormat, srgb ? DirectX::TEX_COMPRESS_SRGB : DirectX::TEX_COMPRESS_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, compressedScratchImage);
			}
			else
			{
				hr = DirectX::Compress(scratchImage.GetImages(), scratchImage.GetImageCount(), scratchImage.GetMetadata(), dxgiFormat, srgb ? DirectX::TEX_COMPRESS_SRGB : DirectX::TEX_COMPRESS_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, compressedScratchImage);
			}
			if (FAILED(hr))
			{
				BB_ERROR("Failed to compress texture.");
				return;
			}
		}
		else
		{
			hr = DirectX::Convert(scratchImage.GetImages(), scratchImage.GetImageCount(), scratchImage.GetMetadata(), dxgiFormat, DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, compressedScratchImage);
			if (FAILED(hr))
			{
				BB_ERROR("Failed to convert texture.");
				return;
			}
		}
		scratchImage = std::move(compressedScratchImage);
	}

	void TextureHelper::EquirectangularToTextureCube(DirectX::ScratchImage& scratchImage, const TextureFormat& uncompressedFormat)
	{
		auto metadata = scratchImage.GetMetadata();
		uint32_t size = std::min(metadata.width, metadata.height);

		uint8_t* temporaryData = BB_MALLOC_ARRAY(uint8_t, scratchImage.GetPixelsSize());
		memcpy(temporaryData, scratchImage.GetPixels(), scratchImage.GetPixelsSize());
		Texture2D* temporaryTexture = Texture2D::Create(metadata.width, metadata.height, 1, uncompressedFormat);
		temporaryTexture->SetData(temporaryData, scratchImage.GetPixelsSize());
		temporaryTexture->Apply();
		GfxTexture* temporaryTextureCube = GfxRenderTexturePool::Get(size, size, 1, 1, uncompressedFormat, TextureDimension::TextureCube, WrapMode::Clamp, FilterMode::Bilinear, true);
		uint32_t blockSize = static_cast<uint32_t>(scratchImage.GetPixelsSize() / (metadata.width * metadata.height));
		size_t dataSize = size * size * 6 * blockSize;

		DirectX::ScratchImage cubeScratchImage = {};
		cubeScratchImage.InitializeCube(static_cast<DXGI_FORMAT>(uncompressedFormat), metadata.width, metadata.height, 1, 1);
	
		if (s_EquirectangularToCubemapMaterial == nullptr)
		{
			s_EquirectangularToCubemapMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/EquirectangularToCubemap.shader")));
		}

		GfxDevice::SetViewCount(6);
		GfxDevice::SetRenderTarget(temporaryTextureCube);
		GfxDevice::SetViewport(0, 0, size, size);
		GfxDevice::SetGlobalTexture(TO_HASH("_EquirectangularTexture"), temporaryTexture->Get());
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_EquirectangularToCubemapMaterial));
		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::SetViewCount(1);
		temporaryTextureCube->GetData(cubeScratchImage.GetPixels());

		Object::Destroy(temporaryTexture);
		GfxRenderTexturePool::Release(temporaryTextureCube);
		scratchImage = std::move(cubeScratchImage);
	}

	void TextureHelper::SlicesToTextureCube(DirectX::ScratchImage& scratchImage)
	{
		auto metadata = scratchImage.GetMetadata();
		const uint32_t size = static_cast<uint32_t>(metadata.height);
		const size_t dataSize = scratchImage.GetPixelsSize();
		const uint32_t bytesPerPixel = DirectX::BitsPerPixel(metadata.format) / 8;

		DirectX::ScratchImage cubeScratchImage = {};
		cubeScratchImage.InitializeCube(metadata.format, size, size, 1, 1);

		size_t rowPitch = size * bytesPerPixel;
		size_t wideRowPitch = rowPitch * 6;
		size_t slicePitch = rowPitch * size;
		size_t wideSlicePitch = slicePitch * 6;
		for (uint32_t i = 0; i < 6; ++i)
		{
			for (uint32_t j = 0; j < size; ++j)
			{
				memcpy(cubeScratchImage.GetPixels() + i * slicePitch + j * rowPitch, scratchImage.GetPixels() + i * rowPitch + j * wideRowPitch, size * bytesPerPixel);
			}
		}
		scratchImage = std::move(cubeScratchImage);
	}

	void TextureHelper::DownscaleTextureCube(GfxTexture* texture, DirectX::ScratchImage& scratchImage)
	{
		auto metadata = scratchImage.GetMetadata();
		const uint32_t bytesPerPixel = 8;
		uint32_t size = std::min(metadata.width, metadata.height);

		if (s_GenerateReflectionMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/GenerateReflection.shader"));
			s_GenerateReflectionMaterial = Material::Create(shader);
		}

		static GfxTexture* temporaryTexture = nullptr;

		if (temporaryTexture == nullptr)
		{
			TextureProperties textureProperties = {};

			textureProperties.width = size;
			textureProperties.height = size;
			textureProperties.depth = 1;
			textureProperties.antiAliasing = 1;
			textureProperties.mipCount = 1;
			textureProperties.format = TextureFormat::R16G16B16A16_Float;
			textureProperties.dimension = TextureDimension::TextureCube;
			textureProperties.wrapMode = WrapMode::Clamp;
			textureProperties.filterMode = FilterMode::Point;
			textureProperties.isReadable = true;
			textureProperties.isRenderTarget = true;
			GfxDevice::CreateTexture(textureProperties, temporaryTexture);
		}

		GfxDevice::SetViewCount(6);
		GfxDevice::SetGlobalTexture(TO_HASH("_SourceTexture"), texture);
		GfxDevice::SetRenderTarget(temporaryTexture, nullptr);
		GfxDevice::SetViewport(0, 0, size, size);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_GenerateReflectionMaterial, 0));
		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::SetViewCount(1);
		temporaryTexture->GetData(scratchImage.GetPixels());

		DirectX::ScratchImage packedImage = {};
		packedImage.Initialize2D(DXGI_FORMAT_R16G16B16A16_FLOAT, size * 6, size, 1, 1);

		size_t rowPitch = size * bytesPerPixel;
		size_t wideRowPitch = rowPitch * 6;
		size_t slicePitch = rowPitch * size;
		size_t wideSlicePitch = slicePitch * 6;
		for (uint32_t i = 0; i < 6; ++i)
		{
			for (uint32_t j = 0; j < size; ++j)
			{
				memcpy(packedImage.GetPixels() + i * rowPitch + j * wideRowPitch, scratchImage.GetPixels() + i * slicePitch + j * rowPitch, size * bytesPerPixel);
			}
		}
		scratchImage = std::move(packedImage);
	}

	struct ReflectionGenerationData
	{
		Vector4 roughness;
	};

	void TextureHelper::ConvoluteSpecularTextureCube(DirectX::ScratchImage& scratchImage)
	{
		auto metadata = scratchImage.GetMetadata();
		uint32_t size = std::min(metadata.width, metadata.height);

		if (s_GenerateReflectionMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/GenerateReflection.shader"));
			s_GenerateReflectionMaterial = Material::Create(shader);
		}

		static GfxBuffer* constantBuffer = nullptr;

		if (constantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(ReflectionGenerationData) * 1;
			constantBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(constantBufferProperties, constantBuffer);
		}

		static GfxTexture* temporaryTexture0 = nullptr;
		static GfxTexture* temporaryTexture1 = nullptr;

		if (temporaryTexture0 == nullptr)
		{
			TextureProperties textureProperties = {};

			textureProperties.width = size;
			textureProperties.height = size;
			textureProperties.depth = 1;
			textureProperties.antiAliasing = 1;
			textureProperties.mipCount = 0;
			textureProperties.format = static_cast<TextureFormat>(metadata.format);
			textureProperties.dimension = TextureDimension::TextureCube;
			textureProperties.wrapMode = WrapMode::Clamp;
			textureProperties.filterMode = FilterMode::Trilinear;
			textureProperties.isWritable = true;
			GfxDevice::CreateTexture(textureProperties, temporaryTexture0);

			textureProperties.mipCount = 6;
			textureProperties.isReadable = true;
			textureProperties.isWritable = false;
			textureProperties.isRenderTarget = true;
			GfxDevice::CreateTexture(textureProperties, temporaryTexture1);
		}

		temporaryTexture0->SetData(scratchImage.GetPixels(), scratchImage.GetPixelsSize());

		GfxDevice::SetViewCount(6);
		uint32_t viewportSize = size;
		for (int i = 0; i < 6; ++i)
		{
			ReflectionGenerationData constants = {};
			constants.roughness = Vector4((float)i / 5, 0, 0, 0);
			constantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
			GfxDevice::SetGlobalBuffer(TO_HASH("_ReflectionGenerationData"), constantBuffer);

			GfxDevice::SetRenderTarget(temporaryTexture1, nullptr, i);
			GfxDevice::SetViewport(0, 0, viewportSize, viewportSize);
			GfxDevice::SetGlobalTexture(TO_HASH("_SourceTexture"), temporaryTexture0);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_GenerateReflectionMaterial, 1));

			viewportSize /= 2;
		}
		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::SetViewCount(1);

		DirectX::ScratchImage cubeScratchImage = {};
		cubeScratchImage.InitializeCube(metadata.format, size, size, 1, 6);

		temporaryTexture1->GetData(cubeScratchImage.GetPixels());
		scratchImage = std::move(cubeScratchImage);
	}
}
