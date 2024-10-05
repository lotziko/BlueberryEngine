#include "bbpch.h"
#include "RegisterIcons.h"

#include "IconDB.h"

#include "Blueberry\Graphics\Mesh.h"
#include "Editor\Assets\Importers\FolderImporter.h"

namespace Blueberry
{
	void RegisterIcons()
	{
		REGISTER_ICON(Object::Type, "assets/icons/FileIcon.png");
		REGISTER_ICON(Mesh::Type, "assets/icons/FbxIcon.png");
		REGISTER_ICON(FolderImporter::Type, "assets/icons/FolderIcon.png");
	}
}
