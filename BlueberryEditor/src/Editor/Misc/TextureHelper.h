#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\Enums.h"
#include "Concrete\DX11\DX11.h"

namespace DirectX
{
	class ScratchImage;
}

namespace Blueberry
{
	class GfxTexture;

	class TextureHelper
	{
	public:
		static void Load(DirectX::ScratchImage& scratchImage, const String& path, const String& extension, const bool& srgb);
		static void Flip(DirectX::ScratchImage& scratchImage);
		static void GenerateMipMaps(DirectX::ScratchImage& scratchImage);
		static void Compress(DirectX::ScratchImage& scratchImage, const TextureFormat& format, const bool& srgb);
		static void EquirectangularToTextureCube(DirectX::ScratchImage& scratchImage, const TextureFormat& uncompressedFormat);
		static void SlicesToTextureCube(DirectX::ScratchImage& scratchImage);
		static void DownscaleTextureCube(GfxTexture* texture, DirectX::ScratchImage& scratchImage);
		static void ConvoluteSpecularTextureCube(DirectX::ScratchImage& scratchImage);
	};
}