#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	void AssetLoader::Initialize(AssetLoader* loader)
	{
		s_Instance = loader;
	}

	void AssetLoader::Load(const Guid& guid)
	{
		return s_Instance->LoadImpl(guid);
	}

	Object* AssetLoader::Load(const Guid& guid, const FileId& fileId)
	{
		return s_Instance->LoadImpl(guid, fileId);
	}

	Object* AssetLoader::Load(const String& path)
	{
		return s_Instance->LoadImpl(path);
	}
}
