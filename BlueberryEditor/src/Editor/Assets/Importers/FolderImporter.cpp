#include "FolderImporter.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(FolderImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(FolderImporter, AssetImporter)
	}

	void FolderImporter::ImportData()
	{
	}
}
