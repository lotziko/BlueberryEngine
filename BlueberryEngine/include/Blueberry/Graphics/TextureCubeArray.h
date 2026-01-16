#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class BB_API TextureCubeArray : public Texture
	{
		OBJECT_DECLARATION(TextureCubeArray)

	public:
		~TextureCubeArray();

		const uint32_t& GetCount();

		void SetData(uint8_t* data, const size_t& dataSize);
		void Apply();

		static TextureCubeArray* Create(const uint32_t& width, const uint32_t& height, const uint32_t& count, const uint32_t& mipCount = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Bilinear);

	private:
		uint32_t m_Count = 0;
	};
}