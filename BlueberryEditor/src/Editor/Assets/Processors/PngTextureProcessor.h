#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\Enums.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"

#include <directxtex\DirectXTex.h>

namespace Blueberry
{
	struct PngTextureProperties
	{
		uint32_t width;
		uint32_t height;
		uint8_t* data;
		size_t dataSize;
		uint32_t mipCount;
		TextureFormat format;
	};

	class PngTextureProcessor
	{
	public:
		PngTextureProcessor() = default;
		~PngTextureProcessor();

		void Load(const String& path, const bool& generateMips);
		void LoadDDS(const String& path);
		void LoadHDR(const String& path);
		void CreateCube(const TextureFormat& format, const uint32_t& width, const uint32_t& height);
		void Compress(const TextureFormat& format, const bool& srgb);
		const PngTextureProperties& GetProperties();
		uint8_t* GetData();
		const size_t GetDataSize();

	private:
		DirectX::ScratchImage m_ScratchImage;
		PngTextureProperties m_Properties;

		static ComPtr<ID3D11Device> s_Device;
	};
}