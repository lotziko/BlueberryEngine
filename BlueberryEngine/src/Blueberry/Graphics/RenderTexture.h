#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class RenderTexture : public Texture
	{
		OBJECT_DECLARATION(RenderTexture)

	public:
		~RenderTexture();

		static RenderTexture* Create(const UINT& width, const UINT& height, const UINT& antiAliasing = 1, const TextureFormat& textureFormat = TextureFormat::R8G8B8A8_UNorm, const WrapMode& wrapMode = WrapMode::Clamp, const FilterMode& filterMode = FilterMode::Linear, const bool& isReadable = false);
	
		static void UpdateTemporary();
		static RenderTexture* GetTemporary(const UINT& width, const UINT& height, const UINT& antiAliasing, const TextureFormat& textureFormat);
		static void ReleaseTemporary(RenderTexture* texture);

	private:
		static std::unordered_map<ObjectId, size_t> s_TemporaryKeys;
		static std::unordered_multimap<size_t, std::pair<RenderTexture*, size_t>> s_TemporaryPool;
	};
}