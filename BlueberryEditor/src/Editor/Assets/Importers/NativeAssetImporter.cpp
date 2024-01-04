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
			BB_INFO(std::string() << "Asset \"" << GetName() << "\" is already imported.");
		}
		else
		{
			YamlSerializer serializer;
			serializer.Deserialize(GetFilePath());

			auto& deserializedObjects = serializer.GetDeserializedObjects();
			for (Object* object : deserializedObjects)
			{
				ObjectDB::AllocateIdToGuid(object, guid);
				AddImportedObject(object);
				object->SetName(GetName());
			}
		}
	}
}
