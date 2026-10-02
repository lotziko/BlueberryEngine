#include "Blueberry\Graphics\GfxTexturePool.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\Time.h"

namespace Blueberry
{
	Dictionary<GfxTexture*, GfxTexturePoolKey> GfxTexturePool::s_TemporaryKeys = {};
	Dictionary<GfxTexturePoolKey, List<GfxTexturePool::TextureData>> GfxTexturePool::s_TemporaryPool = {};
	size_t GfxTexturePool::s_UnusedMemory = 0;

	static size_t s_MaxUnusedMemory = 1024ull * 1024 * 256;

	bool GfxTexturePoolKey::operator==(const GfxTexturePoolKey& other) const
	{
		return first == other.first && second == other.second;
	}

	bool GfxTexturePoolKey::operator!=(const GfxTexturePoolKey& other) const
	{
		return first != other.first || second != other.second;
	}

	void ReturnTextureToPool::operator()(GfxTexture* texture) const
	{
		GfxTexturePool::Release(texture);
	}

	GfxTexturePoolKey GetKey(uint32_t width, uint32_t height, uint32_t depth, TextureUsageFlags usageFlags, uint32_t antiAliasing, uint32_t mipCount, TextureFormat textureFormat, TextureDimension textureDimension, WrapMode wrapMode, FilterMode filterMode)
	{
		GfxTexturePoolKey key;
		key.first = static_cast<uint64_t>(width) | static_cast<uint64_t>(height) << 16 | static_cast<uint64_t>(depth) << 32 | static_cast<uint64_t>(usageFlags) << 40 | static_cast<uint64_t>(antiAliasing) << 48 | static_cast<uint64_t>(mipCount) << 56;
		key.second = static_cast<uint32_t>(textureFormat) | static_cast<uint32_t>(textureDimension) << 8 | static_cast<uint32_t>(wrapMode) << 16 | static_cast<uint32_t>(filterMode) << 24;
		return key;
	}

	void GfxTexturePool::Shutdown()
	{
		for (auto it = s_TemporaryPool.begin(); it != s_TemporaryPool.end(); ++it)
		{
			for (auto& pair : it->second)
			{
				delete pair.texture;
			}
			it->second.clear();
		}
		s_TemporaryPool.clear();
	}

	void GfxTexturePool::Update()
	{
		if (s_UnusedMemory > s_MaxUnusedMemory)
		{
			size_t targetMemory = (s_MaxUnusedMemory * 4) / 5; // Decrease to 80%
			while (s_UnusedMemory > targetMemory)
			{
				ReleaseOldest();
			}
		}
	}

	GfxTexture* GfxTexturePool::Get(const TextureProperties& textureProperties)
	{
		GfxTexturePoolKey key = GetKey(textureProperties.width, textureProperties.height, textureProperties.depth, textureProperties.usageFlags, textureProperties.antiAliasing, textureProperties.mipCount, textureProperties.format, textureProperties.dimension, textureProperties.wrapMode, textureProperties.filterMode);
		GfxTexture* texture = Find(key);

		if (texture == nullptr)
		{
			texture = Allocate(textureProperties);
			s_TemporaryKeys.insert_or_assign(texture, key);
		}
		return texture;
	}

	GfxTexture* GfxTexturePool::Get(uint32_t width, uint32_t height, uint32_t depth, TextureUsageFlags usageFlags, uint32_t antiAliasing, uint32_t mipCount, TextureFormat textureFormat, TextureDimension textureDimension, WrapMode wrapMode, FilterMode filterMode)
	{
		GfxTexturePoolKey key = GetKey(width, height, depth, usageFlags, antiAliasing, mipCount, textureFormat, textureDimension, wrapMode, filterMode);
		GfxTexture* texture = Find(key);

		if (texture == nullptr)
		{
			TextureProperties textureProperties = {};
			textureProperties.width = width;
			textureProperties.height = height;
			textureProperties.depth = depth;
			textureProperties.antiAliasing = antiAliasing;
			textureProperties.mipCount = mipCount;
			textureProperties.format = textureFormat;
			textureProperties.dimension = textureDimension;
			textureProperties.wrapMode = wrapMode;
			textureProperties.filterMode = filterMode;
			textureProperties.usageFlags = usageFlags;

			texture = Allocate(textureProperties);
			s_TemporaryKeys.insert_or_assign(texture, key);
		}
		return texture;
	}

	void GfxTexturePool::Release(GfxTexture* texture)
	{
		if (texture == nullptr)
		{
			return;
		}

		auto it = s_TemporaryKeys.find(texture);
		if (it != s_TemporaryKeys.end())
		{
			TextureData data = { texture, Time::GetFrameCount() };
			auto it1 = s_TemporaryPool.find(it->second);
			if (it1 != s_TemporaryPool.end())
			{
				List<TextureData>& textures = it1->second;
				textures.emplace_back(data);
			}
			else
			{
				List<TextureData> textures = {};
				textures.emplace_back(data);
				s_TemporaryPool.insert({ it->second, textures });
			}
			s_UnusedMemory += texture->GetAllocationSize();
		}
		else
		{
			BB_ERROR("Trying to release non temporary render texture.");
		}
	}

	GfxTexture* GfxTexturePool::Find(const GfxTexturePoolKey& key)
	{
		auto it = s_TemporaryPool.find(key);
		if (it != s_TemporaryPool.end())
		{
			List<TextureData>& textures = it->second;
			if (textures.size() > 0)
			{
				GfxTexture* last = (textures.end() - 1)->texture;
				textures.erase(textures.end() - 1);
				s_UnusedMemory -= last->GetAllocationSize();
				return last;
			}
		}
		return nullptr;
	}

	GfxTexture* GfxTexturePool::Allocate(const TextureProperties& textureProperties)
	{
		GfxTexture* texture = nullptr;
		GfxDevice::CreateTexture(textureProperties, texture);
		return texture;
	}

	bool GfxTexturePool::ReleaseOldest()
	{
		auto oldestBucket = s_TemporaryPool.end();
		size_t oldestIndex = 0;
		size_t oldestFrame = SIZE_MAX;

		for (auto it = s_TemporaryPool.begin(); it != s_TemporaryPool.end(); ++it)
		{
			List<TextureData>& textures = it->second;
			for (size_t i = 0; i < textures.size(); ++i)
			{
				if (oldestBucket == s_TemporaryPool.end() || textures[i].releaseFrame < oldestFrame)
				{
					oldestBucket = it;
					oldestIndex = i;
					oldestFrame = textures[i].releaseFrame;
				}
			}
		}

		if (oldestBucket == s_TemporaryPool.end())
		{
			return false;
		}

		List<TextureData>& textures = oldestBucket->second;
		GfxTexture* texture = textures[oldestIndex].texture;

		textures.erase(textures.begin() + oldestIndex);
		s_TemporaryKeys.erase(texture);
		s_UnusedMemory -= texture->GetAllocationSize();

		if (textures.empty())
		{
			s_TemporaryPool.erase(oldestBucket);
		}

		delete texture;
		return true;
	}
}