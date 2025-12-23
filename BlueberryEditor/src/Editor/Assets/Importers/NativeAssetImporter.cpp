#include "NativeAssetImporter.h"

#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Serialization\YamlSceneSerializer.h"
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

		String path = GetFilePath();
		String extension = String(std::filesystem::path(path).extension().string());
		Serializer* serializer = extension == ".prefab" ? static_cast<Serializer*>(new YamlSceneSerializer()) : (YamlHelper::IsYaml(path) ? static_cast<Serializer*>(new YamlSerializer()) : static_cast<Serializer*>(new BinarySerializer()));
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
			importedObject->SetState(ObjectState::Default);
			if (!mainObjectIsSet)
			{
				importedObject->SetName(GetName());
				SetMainObject(fileId);
				mainObjectIsSet = true;
			}
		}
		delete serializer;
		//BB_INFO("NativeAsset \"" << GetName() << "\" imported.");
	}
}
