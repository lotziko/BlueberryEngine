#include "DefaultImporter.h"

#include "Editor\Assets\AssetDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(DefaultImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(DefaultImporter, AssetImporter)
	}

	void DefaultImporter::ImportData()
	{
	}
}
