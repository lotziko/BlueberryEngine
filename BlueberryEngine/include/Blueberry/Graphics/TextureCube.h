#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class BB_API TextureCube : public Texture
	{
		OBJECT_DECLARATION(TextureCube)
		
	public:
		~TextureCube();

		uint8_t* GetData();
		const size_t GetDataSize();
		void SetData(uint8_t* data, const size_t& dataSize);
		void Apply();

		static TextureCube* Create(const uint32_t& width, const uint32_t& height, const uint32_t& mipCount = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Bilinear, TextureCube* existingTexture = nullptr);
	};
}