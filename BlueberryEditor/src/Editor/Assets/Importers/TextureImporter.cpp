#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Tools\FileHelper.h"
#include "Editor\Assets\AssetDB.h"
#include "stb\stb_image.h"

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
			BB_INFO(std::string() << "Texture \"" << GetName() << "\" is already imported.");
		}
		else if (AssetDB::HasAssetWithGuidInData(guid))
		{
			object = AssetDB::LoadAssetObject<Texture2D>(guid);
			byte* data;
			size_t length;
			FileHelper::Load(data, length, GetTexturePath());
			object->Initialize({ data, length });
			delete[] data;
			BB_INFO(std::string() << "Texture \"" << GetName() << "\" imported from cache.");
		}
		else
		{
			std::string path = GetFilePath();

			stbi_uc* data = nullptr;
			int width, height, channels;

			stbi_set_flip_vertically_on_load(1);
			data = stbi_load(path.c_str(), &width, &height, &channels, 4);
			size_t dataSize = width * height * 4;

			TextureProperties properties;

			properties.width = width;
			properties.height = height;
			properties.data = data;
			properties.dataSize = dataSize;
			properties.isRenderTarget = false;

			object = Texture2D::Create(properties);
			ObjectDB::AllocateIdToGuid(object, guid);
			AssetDB::SaveAssetObjectToCache(object);
			FileHelper::Save(data, dataSize, GetTexturePath());

			stbi_image_free(data);
			BB_INFO(std::string() << "Texture \"" << GetName() << "\" imported and created from: " + path);
		}
		object->SetName(GetName());
		AddImportedObject(object);
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
