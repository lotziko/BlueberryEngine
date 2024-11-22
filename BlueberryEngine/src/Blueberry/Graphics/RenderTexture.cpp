#include "bbpch.h"
#include "RenderTexture.h"

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Texture, RenderTexture)

	std::unordered_map<ObjectId, size_t> RenderTexture::s_TemporaryKeys = {};
	std::unordered_multimap<size_t, std::pair<RenderTexture*, size_t>> RenderTexture::s_TemporaryPool = {};

	RenderTexture::~RenderTexture()
	{
		if (m_Texture != nullptr)
		{
			delete m_Texture;
		}
	}

	RenderTexture* RenderTexture::Create(const uint32_t& width, const uint32_t& height, const uint32_t& antiAliasing, const TextureFormat& textureFormat, const WrapMode& wrapMode, const FilterMode& filterMode, const bool& isReadable)
	{
		RenderTexture* texture = Object::Create<RenderTexture>();
		texture->m_Width = width;
		texture->m_Height = height;
		texture->m_MipCount = 1;
		texture->m_Format = textureFormat;
		texture->m_WrapMode = wrapMode;
		texture->m_FilterMode = filterMode;

		TextureProperties textureProperties = {};

		textureProperties.width = width;
		textureProperties.height = height;
		textureProperties.antiAliasing = antiAliasing;
		textureProperties.mipCount = 1;
		textureProperties.format = textureFormat;
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
			// Release textures older than 5 frames
			if (currentFrame - it->second.second > 5)
			{
				RenderTexture* texture = it->second.first;
				Object::Destroy(texture);
				s_TemporaryKeys.erase(texture->GetObjectId());
				s_TemporaryPool.erase(it);
			}
		}
	}

	RenderTexture* RenderTexture::GetTemporary(const uint32_t& width, const uint32_t& height, const uint32_t& antiAliasing, const TextureFormat& textureFormat)
	{
		// Use 16 bits for width, height and format
		size_t key = width | height << 16 | antiAliasing << 24 | (uint32_t)textureFormat << 40;

		// https://stackoverflow.com/questions/3952476/how-to-remove-a-specific-pair-from-a-c-multimap
		auto iterpair = s_TemporaryPool.equal_range(key);
		auto it = iterpair.first;
		for (; it != iterpair.second; ++it)
		{
			s_TemporaryPool.erase(it);
			return it->second.first;
		}
		
		// Allocate new texture
		RenderTexture* texture = RenderTexture::Create(width, height, antiAliasing, textureFormat);
		s_TemporaryKeys.insert_or_assign(texture->GetObjectId(), key);
		return texture;
	}

	void RenderTexture::ReleaseTemporary(RenderTexture* texture)
	{
		auto it = s_TemporaryKeys.find(texture->GetObjectId());
		if (it != s_TemporaryKeys.end())
		{
			s_TemporaryPool.insert({ it->second, std::make_pair(texture, Time::GetFrameCount()) });
		}
		else
		{
			BB_ERROR("Trying to release non temporary render texture.");
		}
	}
}