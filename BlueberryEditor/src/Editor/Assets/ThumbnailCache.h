#pragma once

namespace Blueberry
{
	class GfxTexture;
	class Texture2D;
	class AssetImporter;

	class ThumbnailCache
	{
	public:
		static Texture2D* GetThumbnail(Object* asset);
		static void Refresh(Object* asset);

	private:
		static Texture2D* Load(Object* asset);
		static void Save(Object* asset, unsigned char* thumbnail);
		static Texture2D* DrawAndSave(Object* asset);

	private:
		static std::unordered_map<ObjectId, Texture2D*> s_Thumbnails;
	};
}