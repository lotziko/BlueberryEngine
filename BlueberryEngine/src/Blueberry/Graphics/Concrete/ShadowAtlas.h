#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Material;
	class GfxTexture;
	class GfxBuffer;
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
			uint32_t sliceIndex;
			uint32_t sliceCount;
		};

	public:
		ShadowAtlas() = default;
		~ShadowAtlas() = default;

		static void Initialize();
		static void Shutdown();

		static void Clear();
		static void Insert(Light* light);
		static void Draw(RenderContext& context, CullingResults& results);
		static const Vector2Int GetSize();

		static GfxTexture* GetAtlasTexture();

	private:
		static void PackRequests();

	private:
		static Material* s_ShadowAtlasMaterial;
		static GfxTexture* s_AtlasTexture;
		static GfxBuffer* s_DepthBlitData;
		static List<ShadowRequest> s_Requests;
	};
}