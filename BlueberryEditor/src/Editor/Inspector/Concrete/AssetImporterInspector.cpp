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
		for (auto& object : assetImporter->GetImportedObjects())
		{
			ObjectItem* item = ObjectDB::IdToObjectItem(object.second);
			if (item != nullptr)
			{
				Object* importedObject = item->object;
				ObjectInspector* importedObjectInspector = ObjectInspectorDB::GetInspector(importedObject->GetType());
				if (importedObjectInspector != nullptr)
				{
					importedObjectInspector->Draw(importedObject);
				}
			}
		}
	}
}
