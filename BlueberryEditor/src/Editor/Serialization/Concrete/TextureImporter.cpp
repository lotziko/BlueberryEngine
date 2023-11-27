#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Editor\Serialization\AssetDB.h"
#include "stb\stb_image.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, TextureImporter)

	void TextureImporter::Serialize(SerializationContext& context, ryml::NodeRef& node)
	{
	}

	void TextureImporter::Deserialize(SerializationContext& context, ryml::NodeRef& node)
	{
	}

	void TextureImporter::ImportData()
	{
		Guid guid = GetGuid();
		// TODO check if dirty too
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			AssetDB::LoadAssetObject<Texture2D>(guid);
		}
		else
		{
			std::string path = GetFilePath();

			stbi_uc* data = nullptr;
			int width, height, channels;

			stbi_set_flip_vertically_on_load(1);
			data = stbi_load(path.c_str(), &width, &height, &channels, 4);

			TextureProperties properties;

			properties.width = width;
			properties.height = height;
			properties.data = data;
			properties.dataSize = width * height * 4;
			properties.isRenderTarget = false;

			Ref<Texture2D> object = AssetDB::CreateAssetObject<Texture2D>(guid, properties);
			AssetDB::SaveAssetObject(object);

			//i'm not sure if asset should stay at memory
			//stbi_image_free(data);
		}
	}
}
