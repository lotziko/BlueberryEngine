#include "bbpch.h"
#include "RenderTexture.h"

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, RenderTexture)

	std::unordered_map<ObjectId, size_t> RenderTexture::s_TemporaryKeys = {};
	std::unordered_map<size_t, std::vector<std::pair<RenderTexture*, size_t>>> RenderTexture::s_TemporaryPool = {};

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
			std::vector<std::pair<RenderTexture*, size_t>>& textures = it->second;
			for (auto it1 = textures.end() - 1; it1 != textures.begin(); --it1)
			{
				// Release textures older than 5 frames
				if (currentFrame - it1->second > 5)
				{
					RenderTexture* texture = it1->first;
					if (texture != nullptr)
					{
						s_TemporaryKeys.erase(texture->GetObjectId());
						Object::Destroy(texture);
						textures.erase(it1);
					}
				}
			}
		}
	}

	RenderTexture* RenderTexture::GetTemporary(const uint32_t& width, const uint32_t& height, const uint32_t& depth, const uint32_t& antiAliasing, const TextureFormat& textureFormat, const TextureDimension& textureDimension)
	{
		// Use 16 bits for width, height and format
		size_t key = (size_t)width | (size_t)height << 16 | (size_t)depth << 32 | (size_t)antiAliasing << 40 | (size_t)textureFormat << 48 | (size_t)textureDimension << 56;

		auto it = s_TemporaryPool.find(key);
		if (it != s_TemporaryPool.end())
		{
			std::vector<std::pair<RenderTexture*, size_t>>& textures = it->second;
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
				std::vector<std::pair<RenderTexture*, size_t>>& textures = it1->second;
				textures.emplace_back(pair);
			}
			else
			{
				std::vector<std::pair<RenderTexture*, size_t>> textures = {};
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