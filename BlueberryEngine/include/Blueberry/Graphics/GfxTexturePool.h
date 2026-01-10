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
		static GfxTexture* Get(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const TextureUsageFlags& usageFlags, const uint32_t& antiAliasing = 1, const uint32_t& mipCount = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const TextureDimension& textureDimension = TextureDimension::Texture2D, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Bilinear);
		static void Release(GfxTexture* texture);

	private:
		static GfxTexture* Find(const GfxTexturePoolKey& key);
		static GfxTexture* Allocate(const TextureProperties& textureProperties);

	private:
		static Dictionary<GfxTexture*, GfxTexturePoolKey> s_TemporaryKeys;
		static Dictionary<GfxTexturePoolKey, List<std::pair<GfxTexture*, size_t>>> s_TemporaryPool;
	};
}