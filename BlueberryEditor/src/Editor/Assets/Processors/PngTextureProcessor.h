#pragma once

#include "Blueberry\Graphics\Enums.h"
#include "directxtex\DirectXTex.h"

namespace Blueberry
{
	struct PngTextureProperties
	{
		UINT width;
		UINT height;
		void* data;
		size_t dataSize;
		UINT mipCount;
		TextureFormat format;
	};

	class PngTextureProcessor
	{
	public:
		PngTextureProcessor() = default;
		~PngTextureProcessor();

		void Load(const std::string& path, const bool& srgb, const bool& generateMips);
		void Compress(const TextureFormat& format);
		const PngTextureProperties& GetProperties();

	private:
		DirectX::ScratchImage m_ScratchImage;
		PngTextureProperties m_Properties;

		static ComPtr<ID3D11Device> s_Device;
	};
}