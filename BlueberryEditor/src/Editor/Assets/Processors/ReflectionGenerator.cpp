#include "ReflectionGenerator.h"

#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Tools\StringConverter.h"

#include "Editor\Misc\TextureHelper.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"

#include <directxtex\DirectXTex.h>

namespace Blueberry
{
	TextureCube* ReflectionGenerator::GenerateReflectionTexture(TextureCube* source)
	{
		DirectX::ScratchImage image = {};
		image.InitializeCube(DXGI_FORMAT_R16G16B16A16_FLOAT, 128, 128, 1, 1);
		TextureHelper::DownscaleTextureCube(source->Get(), image);
		
		String path = EditorSceneManager::GetPath();
		path.replace(path.find(".scene"), 6, "\\ReflectionTexture0.hdr");

		std::filesystem::create_directories(std::filesystem::path(path).parent_path());
		HRESULT hr = DirectX::SaveToHDRFile(*image.GetImages(), WString(path.begin(), path.end()).c_str());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to save texture.");
		}

		AssetDB::Refresh();
		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		TextureImporter* importer = static_cast<TextureImporter*>(AssetDB::GetImporter(relativePath.string().data()));
		importer->SetTextureShape(TextureImporter::TextureShape::TextureCube);
		importer->SetTextureCubeType(TextureImporter::TextureCubeType::Slices);
		importer->SetTextureCubeIBLType(TextureImporter::TextureCubeIBLType::Specular);
		importer->SaveAndReimport();
		Object* obj = ObjectDB::GetObjectFromGuid(importer->GetGuid(), importer->GetMainObject());
		return static_cast<TextureCube*>(obj);
	}
}
