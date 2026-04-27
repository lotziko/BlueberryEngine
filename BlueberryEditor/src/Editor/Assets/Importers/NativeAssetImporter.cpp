#include "NativeAssetImporter.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Serialization\EditorSerializer.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(NativeAssetImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(NativeAssetImporter, AssetImporter)
	}

	void NativeAssetImporter::ImportData()
	{
		Guid guid = GetGuid();
		List<Object*> objects;
		String path = GetFilePath();
		EditorSerializer serializer = {};
		for (auto& object : ObjectDB::GetObjectsFromGuid(guid))
		{
			Object* importedObject = ObjectDB::GetObject(object.second);
			if (importedObject != nullptr)
			{
				serializer.AddObject(importedObject, object.first);
			}
		}
		serializer.SetGuid(guid);
		serializer.Deserialize(path, SerializationFlags::EditorOnly | SerializationFlags::HasHeaders);
		auto& deserializedObjects = serializer.GetDeserializedObjects();

		bool mainObjectIsSet = false;
		for (auto& pair : deserializedObjects)
		{
			Object* importedObject = ObjectDB::GetObject(pair.first);
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
		serializer.FinalizeObjects();
	}
}
