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
		EditorSerializer serializer;
		for (auto& object : ObjectDB::GetObjectsFromGuid(guid))
		{
			Object* importedObject = ObjectDB::GetObject(object.second);
			if (importedObject != nullptr)
			{
				serializer.AddObject(importedObject, object.first);
			}
		}
		serializer.Deserialize(path);
		HashSet<Guid> dependencies;
		serializer.GatherDependencies(dependencies);
		AssetDB::SetDependencies(guid, dependencies);
		auto& deserializedObjects = serializer.GetDeserializedObjects();

		PrefabInstance* instance = GetOrCreateAssetObject<PrefabInstance>(PrefabInstance::Type);
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
		instance->OnCreate();
		AssetDB::SaveAssetObjectsToCache(objects);
	}
}