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
}