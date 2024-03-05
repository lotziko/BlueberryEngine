#include "bbpch.h"
#include "RegisterAssetImporters.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\Importers\ShaderImporter.h"
#include "Editor\Assets\Importers\DefaultImporter.h"
#include "Editor\Assets\Importers\NativeAssetImporter.h"
#include "Editor\Assets\Importers\ModelImporter.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	void RegisterAssetImporters()
	{
		REGISTER_ABSTRACT_CLASS(AssetImporter);
		REGISTER_CLASS(TextureImporter);
		REGISTER_CLASS(ShaderImporter);
		REGISTER_CLASS(DefaultImporter);
		REGISTER_CLASS(NativeAssetImporter);
		REGISTER_CLASS(ModelImporter);

		REGISTER_ASSET_IMPORTER(".png", TextureImporter::Type);
		REGISTER_ASSET_IMPORTER(".shader", ShaderImporter::Type);
		REGISTER_ASSET_IMPORTER(".scene", DefaultImporter::Type);
		REGISTER_ASSET_IMPORTER(".material", NativeAssetImporter::Type);
		REGISTER_ASSET_IMPORTER(".fbx", ModelImporter::Type);
	}
}