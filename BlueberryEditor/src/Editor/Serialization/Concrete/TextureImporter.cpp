#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Editor\Serialization\AssetDB.h"
#include "stb\stb_image.h"
#include "Blueberry\Tools\FileHelper.h"

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
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			object = AssetDB::LoadAssetObject<Texture2D>(guid);
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
			ObjectDB::AssignGuid(object, guid);
			AssetDB::SaveAssetObject(object);

			stbi_image_free(data);
		}
		object->SetName(GetName());
		AddImportedObject(object);
	}
}
