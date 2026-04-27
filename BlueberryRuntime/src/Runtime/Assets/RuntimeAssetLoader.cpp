#include "RuntimeAssetLoader.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Tools\StringHelper.h"

#include <filesystem>

namespace Blueberry
{
	Object* RuntimeAssetLoader::LoadImpl(const Guid& guid, FileId fileId)
	{
		return ObjectDB::GetObjectFromGuid(guid, fileId);
	}

	Object* RuntimeAssetLoader::LoadImpl(const String& path, void* args)
	{
		std::filesystem::path assetPath = path;
		Guid guid = Guid(TO_HASH(StringHelper::ToString(assetPath.filename())), 0);
		FileId fileId = 1;
		return ObjectDB::GetObjectFromGuid(guid, fileId);
	}
}