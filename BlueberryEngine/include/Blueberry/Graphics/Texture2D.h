#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class BB_API Texture2D : public Texture
	{
		OBJECT_DECLARATION(Texture2D)
		
	public:
		~Texture2D();

		void SetData(uint8_t* data, const size_t& dataSize);
		void Apply();

		static Texture2D* Create(const uint32_t& width, const uint32_t& height, const uint32_t& mipCount = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Linear, Texture2D* existingTexture = nullptr);
	};
}