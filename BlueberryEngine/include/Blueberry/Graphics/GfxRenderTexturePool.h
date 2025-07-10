#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	struct GfxRenderTexturePoolKey
	{
		uint64_t first;
		uint8_t second;

		bool operator==(const GfxRenderTexturePoolKey& other) const;
		bool operator!=(const GfxRenderTexturePoolKey& other) const;
	};
}

template <>
struct std::hash<Blueberry::GfxRenderTexturePoolKey>
{
	size_t operator()(const Blueberry::GfxRenderTexturePoolKey& key) const
	{
		return std::hash<uint64_t>()(key.first) ^ (std::hash<uint8_t>()(key.second) << 1);
	}
};

namespace Blueberry
{
	class GfxTexture;

	class GfxRenderTexturePool
	{
	public:
		static void Update();
		static GfxTexture* Get(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const TextureDimension& textureDimension = TextureDimension::Texture2D, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Bilinear, const bool& isReadable = false, const bool& isUnorderedAccess = false);
		static void Release(GfxTexture* texture);

	private:
		static Dictionary<GfxTexture*, GfxRenderTexturePoolKey> s_TemporaryKeys;
		static Dictionary<GfxRenderTexturePoolKey, List<std::pair<GfxTexture*, size_t>>> s_TemporaryPool;
	};
}