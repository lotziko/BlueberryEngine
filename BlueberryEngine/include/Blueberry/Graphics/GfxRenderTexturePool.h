#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	struct GfxRenderTexturePoolKey
	{
		uint64_t first;
		uint16_t second;

		bool operator==(const GfxRenderTexturePoolKey& other) const;
		bool operator!=(const GfxRenderTexturePoolKey& other) const;
	};
}

template <>
struct std::hash<Blueberry::GfxRenderTexturePoolKey>
{
	size_t operator()(const Blueberry::GfxRenderTexturePoolKey& key) const
	{
		return std::hash<uint64_t>()(key.first) ^ (std::hash<uint16_t>()(key.second) << 1);
	}
};

namespace Blueberry
{
	class GfxTexture;

	class GfxRenderTexturePool
	{
	public:
		static void Shutdown();
		static void Update();
		static GfxTexture* Get(const TextureProperties& textureProperties);
		static GfxTexture* Get(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing = 1, const uint32_t& mipCount = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const TextureDimension& textureDimension = TextureDimension::Texture2D, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Bilinear, const bool& isReadable = false, const bool& isUnorderedAccess = false);
		static void Release(GfxTexture* texture);

	private:
		static GfxTexture* Find(const GfxRenderTexturePoolKey& key);
		static GfxTexture* Allocate(const TextureProperties& textureProperties);

	private:
		static Dictionary<GfxTexture*, GfxRenderTexturePoolKey> s_TemporaryKeys;
		static Dictionary<GfxRenderTexturePoolKey, List<std::pair<GfxTexture*, size_t>>> s_TemporaryPool;
	};
}