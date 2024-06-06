#include "bbpch.h"
#include "FolderImporter.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, FolderImporter)

	void FolderImporter::BindProperties()
	{
	}

	std::string FolderImporter::GetIconPath()
	{
		return "assets/icons/FolderIcon.png";
	}

	void FolderImporter::ImportData()
	{
	}
}
