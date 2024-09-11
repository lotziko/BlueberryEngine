#include "bbpch.h"
#include "ThumbnailCache.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\AssetImporter.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Path.h"

namespace Blueberry
{
	std::unordered_map<ObjectId, Texture2D*> ThumbnailCache::s_Thumbnails = std::unordered_map<ObjectId, Texture2D*>();

	#define THUMBNAIL_SIZE 128
	#define THUMBNAIL_DATA_SIZE THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4

	Texture2D* ThumbnailCache::GetThumbnail(AssetImporter* importer)
	{
		if (importer == nullptr)
		{
			return nullptr;
		}

		auto thumbnailIt = s_Thumbnails.find(importer->GetObjectId());
		if (thumbnailIt != s_Thumbnails.end())
		{
			return thumbnailIt->second;
		}

		if (s_ThumbnailRenderTarget == nullptr)
		{
			TextureProperties properties = {};
			properties.width = THUMBNAIL_SIZE;
			properties.height = THUMBNAIL_SIZE;
			properties.isRenderTarget = true;
			properties.isReadable = true;
			properties.format = TextureFormat::R8G8B8A8_UNorm;
			GfxDevice::CreateTexture(properties, s_ThumbnailRenderTarget);
		}

		if (importer->IsClassType(TextureImporter::Type))
		{
			Texture2D* thumbnail = Load(importer);
			if (thumbnail != nullptr)
			{
				s_Thumbnails.insert_or_assign(importer->GetObjectId(), thumbnail);
				return thumbnail;
			}
			importer->ImportDataIfNeeded();
			if (importer->IsImported())
			{
				static size_t blitTextureId = TO_HASH("_BlitTexture");

				unsigned char data[THUMBNAIL_DATA_SIZE];
				Texture2D* importedTexture = (Texture2D*)ObjectDB::GetObject(importer->GetImportedObjects().begin()->second);
				GfxDevice::SetRenderTarget(s_ThumbnailRenderTarget);
				GfxDevice::SetViewport(0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
				GfxDevice::SetGlobalTexture(blitTextureId, importedTexture->Get());
				GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetBlit()));
				GfxDevice::Read(s_ThumbnailRenderTarget, data, Rectangle(0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE));
				GfxDevice::SetRenderTarget(nullptr);

				Texture2D* thumbnail = Texture2D::Create(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
				thumbnail->SetData(data, THUMBNAIL_DATA_SIZE);
				thumbnail->Apply();
				s_Thumbnails.insert_or_assign(importer->GetObjectId(), thumbnail);
				Save(importer, data);
				return thumbnail;
			}
		}
		return nullptr;
	}

	Texture2D* ThumbnailCache::Load(AssetImporter* importer)
	{
		auto thumbnailPath = Path::GetThumbnailCachePath();
		thumbnailPath.append(importer->GetGuid().ToString());
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

	void ThumbnailCache::Save(AssetImporter* importer, unsigned char* thumbnail)
	{
		auto thumbnailPath = Path::GetThumbnailCachePath();
		if (!std::filesystem::exists(thumbnailPath))
		{
			std::filesystem::create_directories(thumbnailPath);
		}
		thumbnailPath.append(importer->GetGuid().ToString());
		FileHelper::Save(thumbnail, THUMBNAIL_DATA_SIZE, thumbnailPath.string());
	}
}
