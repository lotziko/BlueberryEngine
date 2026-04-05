#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	AssetLoader* AssetLoader::s_Instance = nullptr;

	void AssetLoader::Initialize(AssetLoader* loader)
	{
		s_Instance = loader;
	}

	Object* AssetLoader::Load(const Guid& guid, FileId fileId)
	{
		return s_Instance->LoadImpl(guid, fileId);
	}

	Object* AssetLoader::Load(const String& path, void* args)
	{
		return s_Instance->LoadImpl(path, args);
	}

	AssetLoader* AssetLoader::GetInstance()
	{
		return s_Instance;
	}
}
