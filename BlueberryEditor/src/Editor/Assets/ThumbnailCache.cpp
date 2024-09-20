#include "bbpch.h"
#include "ThumbnailCache.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\ThumbnailRenderer.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Path.h"

namespace Blueberry
{
	std::unordered_map<ObjectId, Texture2D*> ThumbnailCache::s_Thumbnails = std::unordered_map<ObjectId, Texture2D*>();

	#define THUMBNAIL_SIZE 128
	#define THUMBNAIL_DATA_SIZE THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4

	Texture2D* ThumbnailCache::GetThumbnail(Object* asset)
	{
		if (asset == nullptr || !ObjectDB::HasGuid(asset))
		{
			return nullptr;
		}

		// Return existing
		auto thumbnailIt = s_Thumbnails.find(asset->GetObjectId());
		if (thumbnailIt != s_Thumbnails.end())
		{
			return thumbnailIt->second;
		}

		// Load and return existing
		Texture2D* thumbnail = Load(asset);
		if (thumbnail != nullptr)
		{
			s_Thumbnails.insert_or_assign(asset->GetObjectId(), thumbnail);
			return thumbnail;
		}

		// Draw and create new
		thumbnail = DrawAndSave(asset);
		if (thumbnail != nullptr)
		{
			s_Thumbnails.insert_or_assign(asset->GetObjectId(), thumbnail);
			return thumbnail;
		}
		return nullptr;
	}

	void ThumbnailCache::Refresh(Object* asset)
	{
		Texture2D* thumbnail = DrawAndSave(asset);
		if (thumbnail != nullptr)
		{
			s_Thumbnails.insert_or_assign(asset->GetObjectId(), thumbnail);
		}
	}

	Texture2D* ThumbnailCache::Load(Object* asset)
	{
		auto thumbnailPath = Path::GetThumbnailCachePath();
		auto pair = ObjectDB::GetGuidAndFileIdFromObject(asset);
		thumbnailPath.append(pair.first.ToString().append(std::to_string(pair.second)));
		if (!std::filesystem::exists(thumbnailPath))
		{
			return nullptr;
		}

		unsigned char* data;
		size_t length;
		FileHelper::Load(data, length, thumbnailPath.string());

		Texture2D* thumbnail = Texture2D::Create(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
		thumbnail->SetData(data, THUMBNAIL_DATA_SIZE);
		thumbnail->Apply();

		return thumbnail;
	}

	void ThumbnailCache::Save(Object* asset, unsigned char* thumbnail)
	{
		auto thumbnailPath = Path::GetThumbnailCachePath();
		auto pair = ObjectDB::GetGuidAndFileIdFromObject(asset);
		if (!std::filesystem::exists(thumbnailPath))
		{
			std::filesystem::create_directories(thumbnailPath);
		}
		thumbnailPath.append(pair.first.ToString().append(std::to_string(pair.second)));
		FileHelper::Save(thumbnail, THUMBNAIL_DATA_SIZE, thumbnailPath.string());
	}

	Texture2D* ThumbnailCache::DrawAndSave(Object* asset)
	{
		if (ThumbnailRenderer::CanDraw(asset->GetType()))
		{
			Guid guid = ObjectDB::GetGuidFromObject(asset);
			AssetImporter* importer = AssetDB::GetImporter(guid);
			if (importer != nullptr)
			{
				importer->ImportDataIfNeeded();

				if (importer->IsImported())
				{
					unsigned char data[THUMBNAIL_DATA_SIZE];
					if (ThumbnailRenderer::Draw(data, THUMBNAIL_SIZE, asset))
					{
						Texture2D* thumbnail = Texture2D::Create(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
						thumbnail->SetData(data, THUMBNAIL_DATA_SIZE);
						thumbnail->Apply();
						Save(asset, data);
						return thumbnail;
					}
				}
			}
		}
		return nullptr;
	}
}
