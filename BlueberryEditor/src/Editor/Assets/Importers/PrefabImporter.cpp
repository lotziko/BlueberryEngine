#include "PrefabImporter.h"

#include "Editor\Serialization\EditorSerializer.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Assets\AssetDB.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Scene\Entity.h"

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
		EditorSerializer serializer = {};
		for (auto& object : ObjectDB::GetObjectsFromGuid(guid))
		{
			Object* importedObject = ObjectDB::GetObject(object.second);
			if (importedObject != nullptr)
			{
				serializer.AddObject(importedObject, object.first);
			}
		}
		serializer.Deserialize(path, SerializationFlags::EditorOnly | SerializationFlags::HasHeaders);
		serializer.FinalizeObjects();
		HashSet<Guid> dependencies;
		serializer.GatherDependencies(dependencies);
		AssetDB::SetDependencies(guid, dependencies);
		auto& deserializedObjects = serializer.GetDeserializedObjects();

		size_t prefabInstanceFileId = TO_HASH("PrefabInstance");
		PrefabInstance* instance = GetOrCreateAssetObject<PrefabInstance>(prefabInstanceFileId);
		objects.push_back(instance);

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
		instance->Initialize();
		AssetDB::SaveAssetObjectsToCache(objects);
	}
}