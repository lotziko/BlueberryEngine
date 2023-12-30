#include "bbpch.h"
#include "RegisterAssetImporters.h"

#include "Blueberry\Core\ClassDB.h"

#include "Editor\Serialization\AssetDB.h"
#include "Editor\Serialization\Concrete\TextureImporter.h"
#include "Editor\Serialization\Concrete\ShaderImporter.h"
#include "Editor\Serialization\Concrete\DefaultImporter.h"

namespace Blueberry
{
	void RegisterAssetImporters()
	{
		REGISTER_ABSTRACT_CLASS(AssetImporter);
		REGISTER_CLASS(TextureImporter);
		REGISTER_CLASS(ShaderImporter);
		REGISTER_CLASS(DefaultImporter);

		REGISTER_ASSET_IMPORTER(".png", TextureImporter::Type);
		REGISTER_ASSET_IMPORTER(".shader", ShaderImporter::Type);
	}
}