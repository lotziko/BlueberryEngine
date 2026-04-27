#include "FolderImporter.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(FolderImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(FolderImporter, AssetImporter)
	}

	bool FolderImporter::IsRequiringReimport() const
	{
		return false;
	}

	bool FolderImporter::IsImportable() const
	{
		return false;
	}

	void FolderImporter::ImportData()
	{
	}
}
