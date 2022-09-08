#include "bbpch.h"
#include "AssetManager.h"

#include "Blueberry\Content\TextureImporter.h"
#include "Blueberry\Content\ShaderImporter.h"

namespace Blueberry
{
	AssetManager::AssetManager()
	{
		Register(new TextureImporter());
		Register(new ShaderImporter());
	}

	void AssetManager::Register(AssetImporter* importer)
	{
		m_Importers.insert({ importer->GetType(), importer });
	}
}