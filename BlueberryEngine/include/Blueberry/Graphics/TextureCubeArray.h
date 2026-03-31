#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class BB_API TextureCubeArray : public Texture
	{
		OBJECT_DECLARATION(TextureCubeArray)

	public:
		~TextureCubeArray();

		uint32_t GetCount() const;

		void SetData(uint8_t* data, size_t dataSize);
		void Apply();

		static TextureCubeArray* Create(uint32_t width, uint32_t height, uint32_t count, uint32_t mipCount = 1, TextureFormat textureFormat = TextureFormat::R8G8B8A8_UNorm, WrapMode wrapMode = WrapMode::Clamp, FilterMode filterMode = FilterMode::Bilinear);

	private:
		uint32_t m_Count = 0;
	};
}