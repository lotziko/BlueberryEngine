#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class BB_API Texture3D : public Texture
	{
		OBJECT_DECLARATION(Texture3D)

	public:
		~Texture3D();

		const uint32_t& GetDepth();

		void SetData(uint8_t* data, const size_t& dataSize);
		void Apply();

		static Texture3D* Create(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Bilinear, Texture3D* existingTexture = nullptr);
	
	private:
		uint32_t m_Depth = 0;
	};
}