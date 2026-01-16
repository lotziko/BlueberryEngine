#include "PrefabImporter.h"

#include "Editor\Serialization\YamlSceneSerializer.h"
#include "Editor\Assets\AssetDB.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(PrefabImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(PrefabImporter, AssetImporter)
	}

	void PrefabImporter::ImportData()
	{
		Guid guid = GetGuid();
		List<Object*> objects;
		String path = GetFilePath();
		YamlSceneSerializer serializer;
		for (auto& object : ObjectDB::GetObjectsFromGuid(guid))
		{
			Object* importedObject = ObjectDB::GetObject(object.second);
			if (importedObject != nullptr)
			{
				serializer.AddObject(importedObject, object.first);
			}
		}
		serializer.Deserialize(path);
		auto& deserializedObjects = serializer.GetDeserializedObjects();

		bool mainObjectIsSet = false;
		for (auto& pair : deserializedObjects)
		{
			Object* importedObject = pair.first;
			objects.push_back(importedObject);
			FileId fileId = pair.second;

			ObjectDB::AllocateIdToGuid(importedObject, guid, fileId);
			importedObject->SetState(ObjectState::Default);
			if (!mainObjectIsSet)
			{
				importedObject->SetName(GetName());
				SetMainObject(fileId);
				mainObjectIsSet = true;
			}
		}
		AssetDB::SaveAssetObjectsToCache(objects);
	}
}