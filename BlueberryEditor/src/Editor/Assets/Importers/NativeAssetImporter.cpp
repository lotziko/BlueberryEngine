#include "bbpch.h"
#include "NativeAssetImporter.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, NativeAssetImporter)

	void NativeAssetImporter::BindProperties()
	{
	}

	void NativeAssetImporter::ImportData()
	{
		Guid guid = GetGuid();

		if (IsImported())
		{
			// TODO think how to deserialize into existing object
			BB_INFO("Asset \"" << GetName() << "\" is already imported.");
			return;
		}
		else
		{
			YamlSerializer serializer;
			for (auto& object : ObjectDB::GetObjectsFromGuid(guid))
			{
				Object* importedObject = ObjectDB::GetObject(object.second);
				if (importedObject != nullptr)
				{
					serializer.AddObject(importedObject, object.first);
				}
			}
			serializer.Deserialize(GetFilePath());

			bool mainObjectIsSet = false;
			auto& deserializedObjects = serializer.GetDeserializedObjects();
			for (auto pair : deserializedObjects)
			{
				Object* importedObject = pair.first;
				FileId fileId = pair.second;

				ObjectDB::AllocateIdToGuid(importedObject, guid, fileId);
				importedObject->SetName(GetName());
				importedObject->SetState(ObjectState::Default);
				if (!mainObjectIsSet)
				{
					SetMainObject(fileId);
					mainObjectIsSet = true;
				}
			}
			BB_INFO("NativeAsset \"" << GetName() << "\" imported.");
		}
	}
}
