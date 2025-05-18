#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class BB_API RenderTexture : public Texture
	{
		OBJECT_DECLARATION(RenderTexture)

	public:
		virtual ~RenderTexture();

		static RenderTexture* Create(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const TextureDimension& textureDimension = TextureDimension::Texture2D, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Linear, const bool& isReadable = false);
	
		static void UpdateTemporary();
		static RenderTexture* GetTemporary(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing, const TextureFormat& textureFormat, const TextureDimension& textureDimension);
		static void ReleaseTemporary(RenderTexture* texture);

	private:
		uint32_t m_Depth = 0;
		uint32_t m_SampleCount = 0;

		static Dictionary<ObjectId, size_t> s_TemporaryKeys;
		static Dictionary<size_t, List<std::pair<RenderTexture*, size_t>>> s_TemporaryPool;
	};
}