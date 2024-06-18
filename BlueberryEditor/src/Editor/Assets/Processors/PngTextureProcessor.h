#pragma once

#include "Blueberry\Graphics\Structs.h"
#include "directxtex\DirectXTex.h"

namespace Blueberry
{
	class PngTextureProcessor
	{
	public:
		PngTextureProcessor() = default;
		~PngTextureProcessor();

		void Load(const std::string& path, const bool& srgb, const bool& generateMips);
		void Compress(const TextureFormat& format);
		const TextureProperties& GetProperties();

	private:
		DirectX::ScratchImage m_ScratchImage;
		TextureProperties m_Properties;

		static ComPtr<ID3D11Device> s_Device;
	};
}