#include "GfxRenderTexturePool.h"

#include "GfxDevice.h"
#include "GfxTexture.h"
#include "..\Core\Time.h"

namespace Blueberry
{
	Dictionary<GfxTexture*, GfxRenderTexturePoolKey> GfxRenderTexturePool::s_TemporaryKeys = {};
	Dictionary<GfxRenderTexturePoolKey, List<std::pair<GfxTexture*, size_t>>> GfxRenderTexturePool::s_TemporaryPool = {};

	bool GfxRenderTexturePoolKey::operator==(const GfxRenderTexturePoolKey& other) const
	{
		return first == other.first && second == other.second;
	}

	bool GfxRenderTexturePoolKey::operator!=(const GfxRenderTexturePoolKey& other) const
	{
		return first != other.first || second != other.second;
	}

	void GfxRenderTexturePool::Update()
	{
		size_t currentFrame = Time::GetFrameCount();
		for (auto it = s_TemporaryPool.begin(); it != s_TemporaryPool.end(); ++it)
		{
			auto& vector = it->second;
			for (int i = static_cast<int>(vector.size() - 1); i >= 0; --i)
			{
				auto& pair = vector[i];
				// Release textures older than 5 frames
				if (currentFrame - pair.second > 5)
				{
					GfxTexture* texture = pair.first;
					if (texture != nullptr)
					{
						s_TemporaryKeys.erase(texture);
						delete texture;
						it->second.erase(vector.begin() + i);
					}
				}
			}
		}
	}

	GfxTexture* GfxRenderTexturePool::Get(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing, const TextureFormat& textureFormat, const TextureDimension& textureDimension, const WrapMode& wrapMode, const FilterMode& filterMode, const bool& isReadable, const bool& isUnorderedAccess)
	{
		GfxRenderTexturePoolKey key;
		key.first = static_cast<uint64_t>(width) | static_cast<uint64_t>(height) << 16 | static_cast<uint64_t>(depth) << 32 | static_cast<uint64_t>(antiAliasing) << 40 | static_cast<uint64_t>(textureFormat) << 48 | static_cast<uint64_t>(textureDimension) << 56;
		key.second = static_cast<uint8_t>(wrapMode) | static_cast<uint8_t>(filterMode) << 3 | static_cast<uint8_t>(isReadable) << 6 | static_cast<uint8_t>(isUnorderedAccess) << 7;

		auto it = s_TemporaryPool.find(key);
		if (it != s_TemporaryPool.end())
		{
			List<std::pair<GfxTexture*, size_t>>& textures = it->second;
			if (textures.size() > 0)
			{
				GfxTexture* last = (textures.end() - 1)->first;
				textures.erase(textures.end() - 1);
				return last;
			}
		}

		TextureProperties textureProperties = {};

		textureProperties.width = width;
		textureProperties.height = height;
		textureProperties.depth = depth;
		textureProperties.antiAliasing = antiAliasing;
		textureProperties.mipCount = 1;
		textureProperties.format = textureFormat;
		textureProperties.dimension = textureDimension;
		textureProperties.wrapMode = wrapMode;
		textureProperties.filterMode = filterMode;
		textureProperties.isRenderTarget = true;
		textureProperties.isReadable = isReadable;
		textureProperties.isUnorderedAccess = isUnorderedAccess;
		
		// Allocate new texture
		GfxTexture* texture;
		GfxDevice::CreateTexture(textureProperties, texture);
		s_TemporaryKeys.insert_or_assign(texture, key);
		return texture;
	}

	void GfxRenderTexturePool::Release(GfxTexture* texture)
	{
		if (texture == nullptr)
		{
			return;
		}

		auto it = s_TemporaryKeys.find(texture);
		if (it != s_TemporaryKeys.end())
		{
			auto pair = std::make_pair(texture, Time::GetFrameCount());
			auto it1 = s_TemporaryPool.find(it->second);
			if (it1 != s_TemporaryPool.end())
			{
				List<std::pair<GfxTexture*, size_t>>& textures = it1->second;
				textures.emplace_back(pair);
			}
			else
			{
				List<std::pair<GfxTexture*, size_t>> textures = {};
				textures.emplace_back(pair);
				s_TemporaryPool.insert({ it->second, textures });
			}
		}
		else
		{
			BB_ERROR("Trying to release non temporary render texture.");
		}
	}
}