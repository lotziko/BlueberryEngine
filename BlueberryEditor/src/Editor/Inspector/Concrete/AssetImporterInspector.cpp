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
		for (auto& object : assetImporter->GetImportedObjects())
		{
			Object* importedObject = ObjectDB::GetObject(object.second);
			if (importedObject != nullptr)
			{
				ObjectInspector* importedObjectInspector = ObjectInspectorDB::GetInspector(importedObject->GetType());
				if (importedObjectInspector != nullptr)
				{
					importedObjectInspector->Draw(importedObject);
				}
			}
		}
	}
}
