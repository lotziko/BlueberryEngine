#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Processors\PngTextureProcessor.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, TextureImporter)

	void TextureImporter::BindProperties()
	{
	}

	void TextureImporter::ImportData()
	{
		Guid guid = GetGuid();
		// TODO check if dirty too

		Texture2D* object;
		if (ObjectDB::HasGuid(guid))
		{
			// TODO think how to deserialize into existing object
			BB_INFO("Texture \"" << GetName() << "\" is already imported.");
			return;
		}
		else 
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			auto objects = AssetDB::LoadAssetObjects(guid);
			if (objects.size() == 1 && objects[0].first->IsClassType(Texture2D::Type))
			{
				object = static_cast<Texture2D*>(objects[0].first);
				byte* data;
				size_t length;
				FileHelper::Load(data, length, GetTexturePath());
				object->Initialize({ data, length });
				delete[] data;
				BB_INFO("Texture \"" << GetName() << "\" imported from cache.");
			}
		}
		else
		{
			std::string path = GetFilePath();

			PngTextureProcessor processor;
			processor.Load(path);
			TextureProperties properties = processor.GetProperties();

			object = Texture2D::Create(properties);
			ObjectDB::AllocateIdToGuid(object, guid, 1);
			AssetDB::SaveAssetObjectsToCache(std::vector<Object*> { object });
			FileHelper::Save((byte*)properties.data, properties.dataSize, GetTexturePath());

			BB_INFO("Texture \"" << GetName() << "\" imported and created from: " + path);
		}
		object->SetName(GetName());
		AddImportedObject(object, 1);
	}

	std::string TextureImporter::GetTexturePath()
	{
		std::filesystem::path dataPath = Path::GetTextureCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		dataPath.append(GetGuid().ToString().append(".texture"));
		return dataPath.string();
	}
}
