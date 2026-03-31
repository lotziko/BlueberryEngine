#include "DefaultImporter.h"

#include "Editor\Assets\AssetDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(DefaultImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(DefaultImporter, AssetImporter)
	}

	bool DefaultImporter::IsRequiringReimport() const
	{
		return false;
	}

	bool DefaultImporter::IsImportable() const
	{
		return false;
	}

	void DefaultImporter::ImportData()
	{
	}
}
