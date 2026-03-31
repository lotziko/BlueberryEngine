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
		size_t GetDataSize() const;

		void Initialize(uint32_t width, uint32_t height, uint32_t mipCount = 1, TextureFormat textureFormat = TextureFormat::R8G8B8A8_UNorm);
		void SetData(uint8_t* data, size_t dataSize);
		void Apply();

		static TextureCube* Create(uint32_t width, uint32_t height, uint32_t mipCount = 1, TextureFormat textureFormat = TextureFormat::R8G8B8A8_UNorm, WrapMode wrapMode = WrapMode::Clamp, FilterMode filterMode = FilterMode::Bilinear);
	};
}