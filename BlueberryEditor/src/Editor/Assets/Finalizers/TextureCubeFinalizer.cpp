#include "TextureCubeFinalizer.h"

#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\Importers\TextureImporter.h"

namespace Blueberry
{
	void TextureCubeFinalizer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		TextureCube* texture = static_cast<TextureCube*>(object);
		String texturePath = TextureImporter::GetTexturePath(guid);
		uint8_t* data;
		size_t length;
		FileHelper::Load(data, length, texturePath);
		texture->SetData(data, length);
		texture->Apply();
	}
}
