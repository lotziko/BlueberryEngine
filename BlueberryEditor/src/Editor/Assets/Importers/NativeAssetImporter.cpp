#include "NativeAssetImporter.h"

#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Blueberry\Serialization\BinarySerializer.h"
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

		if (IsImported())
		{
			// TODO think how to deserialize into existing object
			BB_INFO("Asset \"" << GetName() << "\" is already imported.");
			return;
		}
		else
		{
			String path = GetFilePath();
			Serializer* serializer = YamlHelper::IsYaml(path) ? static_cast<Serializer*>(new YamlSerializer()) : static_cast<Serializer*>(new BinarySerializer());
			for (auto& object : ObjectDB::GetObjectsFromGuid(guid))
			{
				Object* importedObject = ObjectDB::GetObject(object.second);
				if (importedObject != nullptr)
				{
					serializer->AddObject(importedObject, object.first);
				}
			}
			serializer->Deserialize(path);
			auto& deserializedObjects = serializer->GetDeserializedObjects();
			
			bool mainObjectIsSet = false;
			for (auto& pair : deserializedObjects)
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
			delete serializer;
			//BB_INFO("NativeAsset \"" << GetName() << "\" imported.");
		}
	}
}
