#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxTexture;
	class Light;
	class RenderContext;
	struct CullingResults;

	class ShadowAtlas
	{
	public:
		BB_OVERRIDE_NEW_DELETE;

		struct ShadowRequest
		{
			uint32_t size;
			uint32_t offsetX;
			uint32_t offsetY;
			Light* light;
			uint8_t sliceIndex;
		};

	public:
		ShadowAtlas() = default;
		ShadowAtlas(const uint32_t& width, const uint32_t& height, const uint32_t& maxLightCount);
		~ShadowAtlas();

		void Clear();
		void Insert(Light* light, const uint32_t& size, const uint8_t& sliceCount);
		void Draw(RenderContext& context, CullingResults& results);
		const Vector2Int& GetSize();

		GfxTexture* GetAtlasTexture();

	private:
		void PackRequests();

	private:
		GfxTexture* m_AtlasTexture = nullptr;
		ShadowRequest* m_Requests;
		Vector2Int m_Size;
		uint32_t m_MaxLightCount;
		uint32_t m_RequestCount;
	};
}