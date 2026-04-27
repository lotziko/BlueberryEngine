#include "RegisterAssetImporters.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\Importers\ShaderImporter.h"
#include "Editor\Assets\Importers\ComputeShaderImporter.h"
#include "Editor\Assets\Importers\DefaultImporter.h"
#include "Editor\Assets\Importers\NativeAssetImporter.h"
#include "Editor\Assets\Importers\PrefabImporter.h"
#include "Editor\Assets\Importers\ModelImporter.h"
#include "Editor\Assets\Importers\AudioImporter.h"
#include "Editor\Assets\Importers\FolderImporter.h"
#include "Editor\Assets\Importers\FontImporter.h"
#include "Editor\Assets\Importers\UIDocumentImporter.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	void RegisterAssetImporters()
	{
		REGISTER_ABSTRACT_CLASS(AssetImporter);
		REGISTER_CLASS(TextureImporter);
		REGISTER_CLASS(ShaderImporter);
		REGISTER_CLASS(ComputeShaderImporter);
		REGISTER_CLASS(DefaultImporter);
		REGISTER_CLASS(NativeAssetImporter);
		REGISTER_CLASS(PrefabImporter);
		REGISTER_DATA_CLASS(ModelMaterialData);
		REGISTER_DATA_CLASS(ModelAnimationClipData);
		REGISTER_CLASS(ModelImporter);
		REGISTER_CLASS(AudioImporter);
		REGISTER_CLASS(FolderImporter);
		REGISTER_CLASS(FontImporter);
		REGISTER_CLASS(UIDocumentImporter);

		REGISTER_ASSET_IMPORTER(".png", TextureImporter::Type);
		REGISTER_ASSET_IMPORTER(".tif", TextureImporter::Type);
		REGISTER_ASSET_IMPORTER(".tiff", TextureImporter::Type);
		REGISTER_ASSET_IMPORTER(".dds", TextureImporter::Type);
		REGISTER_ASSET_IMPORTER(".hdr", TextureImporter::Type);
		REGISTER_ASSET_IMPORTER(".shader", ShaderImporter::Type);
		REGISTER_ASSET_IMPORTER(".compute", ComputeShaderImporter::Type);
		REGISTER_ASSET_IMPORTER(".scene", DefaultImporter::Type);
		REGISTER_ASSET_IMPORTER(".material", NativeAssetImporter::Type);
		REGISTER_ASSET_IMPORTER(".prefab", PrefabImporter::Type);
		REGISTER_ASSET_IMPORTER(".animgraph", NativeAssetImporter::Type);
		REGISTER_ASSET_IMPORTER(".asset", NativeAssetImporter::Type);
		REGISTER_ASSET_IMPORTER(".fbx", ModelImporter::Type);
		REGISTER_ASSET_IMPORTER(".mp3", AudioImporter::Type);
		REGISTER_ASSET_IMPORTER(".wav", AudioImporter::Type);
		REGISTER_ASSET_IMPORTER(".ttf", FontImporter::Type);
		REGISTER_ASSET_IMPORTER(".otf", FontImporter::Type);
		REGISTER_ASSET_IMPORTER(".rml", UIDocumentImporter::Type);
		REGISTER_ASSET_IMPORTER("", FolderImporter::Type);
	}
}