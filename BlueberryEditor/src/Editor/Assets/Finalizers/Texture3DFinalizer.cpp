#include "Texture3DFinalizer.h"

#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\Importers\TextureImporter.h"

namespace Blueberry
{
	void Texture3DFinalizer::Finalize(Object* object, const Guid& guid, const FileId& fileId)
	{
		Texture3D* texture = static_cast<Texture3D*>(object);
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
