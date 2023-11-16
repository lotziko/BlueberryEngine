#include "bbpch.h"
#include "TextureImporter.h"

#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Editor\Serialization\AssetDB.h"
#include "stb\stb_image.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, TextureImporter)

	void TextureImporter::SerializeMeta(YAML::Emitter& out)
	{
		out << YAML::Key << "Test" << YAML::Value << "Test";
	}

	void TextureImporter::DeserializeMeta(YAML::Node& in)
	{
	}

	void TextureImporter::ImportData()
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
		properties.isRenderTarget = false;

		AssetDB::CreateAssetObject<Texture2D>(GetGuid(), properties);

		stbi_image_free(data);
	}
}
