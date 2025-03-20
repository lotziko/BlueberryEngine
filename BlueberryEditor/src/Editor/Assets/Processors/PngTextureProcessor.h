#pragma once

#include "Blueberry\Graphics\Enums.h"
#include "directxtex\DirectXTex.h"

namespace Blueberry
{
	struct PngTextureProperties
	{
		uint32_t width;
		uint32_t height;
		void* data;
		size_t dataSize;
		uint32_t mipCount;
		TextureFormat format;
	};

	class PngTextureProcessor
	{
	public:
		PngTextureProcessor() = default;
		~PngTextureProcessor();

		void Load(const std::string& path, const bool& srgb, const bool& generateMips);
		void LoadDDS(const std::string& path);
		void LoadHDR(const std::string& path);
		void Compress(const TextureFormat& format);
		const PngTextureProperties& GetProperties();

	private:
		DirectX::ScratchImage m_ScratchImage;
		PngTextureProperties m_Properties;

		static ComPtr<ID3D11Device> s_Device;
	};
}