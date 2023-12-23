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

	const std::vector<ObjectId>& AssetImporter::GetImportedObjects()
	{
		return m_ImportedObjects;
	}

	void AssetImporter::Save()
	{
		YamlMetaSerializer serializer;
		serializer.SetGuid(m_Guid);
		serializer.AddObject(this);
		serializer.Serialize(m_MetaPath);
	}

	AssetImporter* AssetImporter::Create(const size_t& type, const std::filesystem::path& path, const std::filesystem::path& metaPath)
	{
		auto info = ClassDB::GetInfo(type);
		AssetImporter* importer = (AssetImporter*)info.createInstance();
		importer->m_Guid = Guid::Create();
		importer->m_Path = path.string();
		importer->m_MetaPath = metaPath.string();
		importer->m_Name = path.stem().string();
		importer->Save();
		importer->ImportData();
		return importer;
	}

	AssetImporter* AssetImporter::Load(const std::filesystem::path& path, const std::filesystem::path& metaPath)
	{
		YamlMetaSerializer serializer;
		serializer.Deserialize(metaPath.string());

		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			AssetImporter* importer = (AssetImporter*)deserializedObjects[0];
			importer->m_Guid = serializer.GetGuid();
			importer->m_Path = path.string();
			importer->m_MetaPath = metaPath.string();
			importer->m_Name = path.stem().string();
			importer->ImportData();
			return importer;
		}
		return nullptr;
	}

	void AssetImporter::BindProperties()
	{
	}

	void AssetImporter::AddImportedObject(Object* object)
	{
		m_ImportedObjects.emplace_back(object->GetObjectId());
	}
}
