#include "bbpch.h"

#include "AssetImporter.h"
#include "Blueberry\Core\ClassDB.h"

#include "Editor\Assets\AssetDB.h"
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

	std::string AssetImporter::GetFilePath()
	{
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(m_RelativePath);
		return dataPath.string();
	}

	std::string AssetImporter::GetMetaFilePath()
	{
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(m_RelativeMetaPath);
		return dataPath.string();
	}

	const std::string& AssetImporter::GetRelativeFilePath()
	{
		return m_RelativePath;
	}

	const std::string& AssetImporter::GetRelativeMetaFilePath()
	{
		return m_RelativeMetaPath;
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
		serializer.Serialize(GetMetaFilePath());
	}

	AssetImporter* AssetImporter::Create(const size_t& type, const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath)
	{
		auto info = ClassDB::GetInfo(type);
		AssetImporter* importer = (AssetImporter*)info.createInstance();
		importer->m_Guid = Guid::Create();
		importer->m_RelativePath = relativePath.string();
		importer->m_RelativeMetaPath = relativeMetaPath.string();
		importer->m_Name = relativePath.stem().string();
		importer->Save();
		importer->ImportData();
		return importer;
	}

	AssetImporter* AssetImporter::Load(const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath)
	{
		YamlMetaSerializer serializer;
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(relativeMetaPath.string());
		serializer.Deserialize(dataPath.string());

		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			AssetImporter* importer = (AssetImporter*)deserializedObjects[0];
			importer->m_Guid = serializer.GetGuid();
			importer->m_RelativePath = relativePath.string();
			importer->m_RelativeMetaPath = relativeMetaPath.string();
			importer->m_Name = relativePath.stem().string();
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
		AssetDB::AddObjectToAsset(object, GetRelativeFilePath());
	}
}
