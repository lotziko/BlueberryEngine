#include "bbpch.h"
#include "RenderTexture.h"

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, RenderTexture)

	Dictionary<ObjectId, size_t> RenderTexture::s_TemporaryKeys = {};
	Dictionary<size_t, List<std::pair<RenderTexture*, size_t>>> RenderTexture::s_TemporaryPool = {};

	RenderTexture::~RenderTexture()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	RenderTexture* RenderTexture::Create(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing, const TextureFormat& textureFormat, const TextureDimension& textureDimension, const WrapMode& wrapMode, const FilterMode& filterMode, const bool& isReadable)
	{
		RenderTexture* texture = Object::Create<RenderTexture>();
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_Depth = depth;
		texture->m_MipCount = 1;
		texture->m_Format = textureFormat;
		texture->m_Dimension = textureDimension;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;
		texture->m_SampleCount = antiAliasing;

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

		GfxDevice::CreateTexture(textureProperties, texture->m_Texture);

		return texture;
	}

	void RenderTexture::UpdateTemporary()
	{
		size_t currentFrame = Time::GetFrameCount();
		for (auto it = s_TemporaryPool.begin(); it != s_TemporaryPool.end(); ++it)
		{
			auto& vector = it->second;
			for (int i = vector.size() - 1; i >= 0; --i)
			{
				auto& pair = vector[i];
				// Release textures older than 5 frames
				if (currentFrame - pair.second > 5)
				{
					RenderTexture* texture = pair.first;
					if (texture != nullptr)
					{
						s_TemporaryKeys.erase(texture->GetObjectId());
						Object::Destroy(texture);
						it->second.erase(vector.begin() + i);
					}
				}
			}
		}
	}

	RenderTexture* RenderTexture::GetTemporary(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing, const TextureFormat& textureFormat, const TextureDimension& textureDimension)
	{
		// Use 16 bits for width, height and format
		size_t key = static_cast<size_t>(width) | static_cast<size_t>(height) << 16 | static_cast<size_t>(depth) << 32 | static_cast<size_t>(antiAliasing) << 40 | static_cast<size_t>(textureFormat) << 48 | static_cast<size_t>(textureDimension) << 56;

		auto it = s_TemporaryPool.find(key);
		if (it != s_TemporaryPool.end())
		{
			List<std::pair<RenderTexture*, size_t>>& textures = it->second;
			if (textures.size() > 0)
			{
				RenderTexture* last = (textures.end() - 1)->first; 
				textures.erase(textures.end() - 1);
				return last;
			}
		}
		
		// Allocate new texture
		RenderTexture* texture = RenderTexture::Create(width, height, depth, antiAliasing, textureFormat, textureDimension);
		s_TemporaryKeys.insert_or_assign(texture->GetObjectId(), key);
		return texture;
	}

	void RenderTexture::ReleaseTemporary(RenderTexture* texture)
	{
		if (texture == nullptr)
		{
			return;
		}

		auto it = s_TemporaryKeys.find(texture->GetObjectId());
		if (it != s_TemporaryKeys.end())
		{
			auto pair = std::make_pair(texture, Time::GetFrameCount());
			auto it1 = s_TemporaryPool.find(it->second);
			if (it1 != s_TemporaryPool.end())
			{
				List<std::pair<RenderTexture*, size_t>>& textures = it1->second;
				textures.emplace_back(pair);
			}
			else
			{
				List<std::pair<RenderTexture*, size_t>> textures = {};
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