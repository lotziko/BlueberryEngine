#pragma once

namespace Blueberry
{
	class GfxTexture;
	class Texture2D;
	class AssetImporter;

	class ThumbnailCache
	{
	public:
		static Texture2D* GetThumbnail(AssetImporter* importer);

	private:
		static Texture2D* Load(AssetImporter* importer);
		static void Save(AssetImporter* importer, unsigned char* thumbnail);

	private:
		static inline GfxTexture* s_ThumbnailRenderTarget = nullptr;
		static std::unordered_map<ObjectId, Texture2D*> s_Thumbnails;
	};
}