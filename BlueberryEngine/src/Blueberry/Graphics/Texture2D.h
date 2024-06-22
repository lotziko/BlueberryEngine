#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class Texture2D : public Texture
	{
		OBJECT_DECLARATION(Texture2D)
		
	public:
		~Texture2D();

		void SetData(byte* data, const size_t& dataSize);
		void Apply();

		static Texture2D* Create(const UINT& width, const UINT& height, const UINT& mipCount = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Linear, Texture2D* existingTexture = nullptr);

		static void BindProperties();
	};
}