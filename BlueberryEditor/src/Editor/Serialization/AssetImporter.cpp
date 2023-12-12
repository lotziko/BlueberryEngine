#include "bbpch.h"

#include "AssetImporter.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Serialization\YamlMetaSerializer.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, AssetImporter)

	const Guid& AssetImporter::GetGuid()
	{
		return m_Guid;
	}

	const std::string& AssetImporter::GetFilePath()
	{
		return m_Path;
	}

	const std::string& AssetImporter::GetMetaFilePath()
	{
		return m_MetaPath;
	}

	void AssetImporter::Save()
	{
		YamlMetaSerializer serializer;
		serializer.SetGuid(m_Guid);
		serializer.AddObject(this);
		serializer.Serialize(m_MetaPath);
	}

	Ref<AssetImporter> AssetImporter::Create(const size_t& type, const std::string& path, const std::string& metaPath)
	{
		auto info = ClassDB::GetInfo(type);
		Ref<AssetImporter> importer = std::dynamic_pointer_cast<AssetImporter>(info.createInstance());
		importer->m_Guid = Guid::Create();
		importer->m_Path = path;
		importer->m_MetaPath = metaPath;
		importer->Save();
		importer->ImportData();
		return importer;
	}

	Ref<AssetImporter> AssetImporter::Load(const std::string& path, const std::string& metaPath)
	{
		YamlMetaSerializer serializer;
		serializer.Deserialize(metaPath);

		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			Ref<AssetImporter> importer = std::dynamic_pointer_cast<AssetImporter>(deserializedObjects[0]);
			importer->m_Guid = serializer.GetGuid();
			importer->m_Path = path;
			importer->m_MetaPath = metaPath;
			importer->ImportData();
			return importer;
		}
		return nullptr;
	}

	void AssetImporter::BindProperties()
	{
	}
}
