#include "bbpch.h"
#include "RegisterAssetImporters.h"

#include "Blueberry\Core\ClassDB.h"

#include "Editor\Serialization\AssetDB.h"
#include "Editor\Serialization\Concrete\TextureImporter.h"

namespace Blueberry
{
	void RegisterAssetImporters()
	{
		REGISTER_ABSTRACT_CLASS(AssetImporter);
		REGISTER_CLASS(TextureImporter);

		REGISTER_ASSET_IMPORTER(".png", TextureImporter::Type);
	}
}