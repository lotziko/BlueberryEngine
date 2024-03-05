#include "bbpch.h"
#include "AssetLoader.h"

namespace Blueberry
{
	void AssetLoader::Initialize(AssetLoader* loader)
	{
		s_Instance = loader;
	}

	Object* AssetLoader::Load(const Guid& guid, const FileId& fileId)
	{
		return s_Instance->LoadImpl(guid, fileId);
	}

	Object* AssetLoader::Load(const std::string& path)
	{
		return s_Instance->LoadImpl(path);
	}
}
