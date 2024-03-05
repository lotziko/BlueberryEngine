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

		if (ObjectDB::HasGuid(guid))
		{
			// TODO think how to deserialize into existing object
			BB_INFO("Asset \"" << GetName() << "\" is already imported.");
			return;
		}
		else
		{
			YamlSerializer serializer;
			serializer.Deserialize(GetFilePath());

			auto& deserializedObjects = serializer.GetDeserializedObjects();
			for (auto pair : deserializedObjects)
			{
				ObjectDB::AllocateIdToGuid(pair.first, guid, pair.second);
				AddImportedObject(pair.first, pair.second);
				pair.first->SetName(GetName());
			}
			BB_INFO("NativeAsset \"" << GetName() << "\" imported.");
		}
	}
}
