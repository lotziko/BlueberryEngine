#include "bbpch.h"
#include "AssetImporterInspector.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Blueberry\Core\ObjectDB.h"
#include "imgui\imgui.h"

namespace Blueberry
{
	void AssetImporterInspector::Draw(Object* object)
	{
		AssetImporter* assetImporter = static_cast<AssetImporter*>(object);
		assetImporter->ImportDataIfNeeded();
		Object* mainObject = ObjectDB::GetObjectFromGuid(assetImporter->GetGuid(), assetImporter->GetMainObject());
		if (mainObject != nullptr)
		{
			ObjectInspector* importedObjectInspector = ObjectInspectorDB::GetInspector(mainObject->GetType());
			if (importedObjectInspector != nullptr)
			{
				importedObjectInspector->Draw(mainObject);
			}
		}
	}
}
