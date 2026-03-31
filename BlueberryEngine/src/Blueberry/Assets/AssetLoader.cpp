#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	AssetLoader* AssetLoader::s_Instance = nullptr;

	void AssetLoader::Initialize(AssetLoader* loader)
	{
		s_Instance = loader;
	}

	void AssetLoader::Load(const Guid& guid)
	{
		return s_Instance->LoadImpl(guid);
	}

	Object* AssetLoader::Load(const Guid& guid, FileId fileId)
	{
		return s_Instance->LoadImpl(guid, fileId);
	}

	Object* AssetLoader::Load(const String& path, void* args)
	{
		return s_Instance->LoadImpl(path, args);
	}
}
