#include "DefaultImporter.h"

#include "Editor\Assets\AssetDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(DefaultImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(DefaultImporter, AssetImporter)
	}

	const bool DefaultImporter::IsRequiringReimport()
	{
		return false;
	}

	void DefaultImporter::ImportData()
	{
	}
}
