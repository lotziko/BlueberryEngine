#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	struct GfxTexturePoolKey
	{
		uint64_t first;
		uint32_t second;

		bool operator==(const GfxTexturePoolKey& other) const;
		bool operator!=(const GfxTexturePoolKey& other) const;
	};
}

template <>
struct std::hash<Blueberry::GfxTexturePoolKey>
{
	size_t operator()(const Blueberry::GfxTexturePoolKey& key) const
	{
		return std::hash<uint64_t>()(key.first) ^ (std::hash<uint32_t>()(key.second) << 1);
	}
};

namespace Blueberry
{
	class GfxTexture;

	class GfxTexturePool
	{
	public:
		static void Shutdown();
		static void Update();
		static GfxTexture* Get(const TextureProperties& textureProperties);
		static GfxTexture* Get(uint32_t width, uint32_t height, uint32_t depth, TextureUsageFlags usageFlags, uint32_t antiAliasing = 1, uint32_t mipCount = 1, TextureFormat textureFormat = TextureFormat::R8G8B8A8_UNorm, TextureDimension textureDimension = TextureDimension::Texture2D, WrapMode wrapMode = WrapMode::Clamp, FilterMode filterMode = FilterMode::Bilinear);
		static void Release(GfxTexture* texture);

	private:
		static GfxTexture* Find(const GfxTexturePoolKey& key);
		static GfxTexture* Allocate(const TextureProperties& textureProperties);

	private:
		static Dictionary<GfxTexture*, GfxTexturePoolKey> s_TemporaryKeys;
		static Dictionary<GfxTexturePoolKey, List<std::pair<GfxTexture*, size_t>>> s_TemporaryPool;
	};
}