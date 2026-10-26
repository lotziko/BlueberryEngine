#include "ThumbnailCache.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Blueberry\Threading\JobSystem.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\ThumbnailRenderer.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Path.h"

namespace Blueberry
{
	Dictionary<ObjectId, Texture2D*> ThumbnailCache::s_Thumbnails = {};
	HashSet<ObjectId> ThumbnailCache::s_InvalidAssets = {};

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

	void ThumbnailCache::Release(Object* asset)
	{
		ObjectId id = asset->GetObjectId();
		s_InvalidAssets.insert(id);
		auto it = s_Thumbnails.find(id);
		if (it != s_Thumbnails.end())
		{
			Object::Destroy(it->second);	// TODO pool
			s_Thumbnails.erase(it);
		}
	}

	Texture2D* ThumbnailCache::Load(Object* asset)
	{
		if (s_InvalidAssets.count(asset->GetObjectId()) > 0)
		{
			return nullptr;
		}

		std::filesystem::path thumbnailPath = Path::GetThumbnailCachePath();
		auto pair = ObjectDB::GetGuidAndFileIdFromObject(asset);
		thumbnailPath.append(pair.first.ToString().append(std::to_string(pair.second)));
		if (!std::filesystem::exists(thumbnailPath))
		{
			return nullptr;
		}
		
		Texture2D* thumbnail = Texture2D::Create(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
		thumbnail->SetState(ObjectState::Loading);
		JobSystem::Execute([thumbnailPath, thumbnail]()
		{
			List<uint8_t> data = FileHelper::LoadBinary(StringHelper::ToString(thumbnailPath));
			thumbnail->SetData(data.data(), data.size());
			JobSystem::PostMain([thumbnail]()
			{
				thumbnail->Apply();
				thumbnail->SetState(ObjectState::Default);
			});
		});
		return thumbnail;
	}

	void ThumbnailCache::Save(Object* asset, unsigned char* thumbnail)
	{
		s_InvalidAssets.erase(asset->GetObjectId());
		std::filesystem::path thumbnailPath = Path::GetThumbnailCachePath();
		auto pair = ObjectDB::GetGuidAndFileIdFromObject(asset);
		if (!std::filesystem::exists(thumbnailPath))
		{
			std::filesystem::create_directories(thumbnailPath);
		}
		thumbnailPath.append(pair.first.ToString().append(std::to_string(pair.second)));
		FileHelper::Save(thumbnail, THUMBNAIL_DATA_SIZE, StringHelper::ToString(thumbnailPath));
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
					List<uint8_t> data(THUMBNAIL_DATA_SIZE);
					if (ThumbnailRenderer::Draw(data.data(), THUMBNAIL_SIZE, asset))
					{
						Texture2D* thumbnail = Texture2D::Create(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
						thumbnail->SetData(data.data(), THUMBNAIL_DATA_SIZE);
						thumbnail->Apply();
						Save(asset, data.data());
						return thumbnail;
					}
				}
			}
		}
		return nullptr;
	}
}
