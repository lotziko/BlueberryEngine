#include "Texture2DFinalizer.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\Importers\TextureImporter.h"

namespace Blueberry
{
	void Texture2DFinalizer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		Texture2D* texture = static_cast<Texture2D*>(object);
		String texturePath = TextureImporter::GetTexturePath(guid);
		if (std::filesystem::exists(texturePath))
		{
			uint8_t* data;
			size_t length;
			FileHelper::Load(data, length, texturePath);
			texture->SetData(data, length);
		}
		texture->Apply();
	}
}
