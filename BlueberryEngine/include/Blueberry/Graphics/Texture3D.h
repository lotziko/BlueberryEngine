#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class BB_API Texture3D : public Texture
	{
		OBJECT_DECLARATION(Texture3D)

	public:
		~Texture3D();

		uint32_t GetDepth() const;

		void Initialize(uint32_t width, uint32_t height, uint32_t depth, TextureFormat textureFormat = TextureFormat::R8G8B8A8_UNorm);
		void SetData(uint8_t* data, size_t dataSize);
		void Apply();

		static Texture3D* Create(uint32_t width, uint32_t height, uint32_t depth, TextureFormat textureFormat = TextureFormat::R8G8B8A8_UNorm, WrapMode wrapMode = WrapMode::Clamp, FilterMode filterMode = FilterMode::Bilinear);
	
	private:
		uint32_t m_Depth = 0;
	};
}