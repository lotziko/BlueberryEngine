#include "bbpch.h"

#include "AssetImporter.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Assets\AssetLoader.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Serialization\YamlMetaSerializer.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, Object)
	{
		DEFINE_BASE_FIELDS(AssetImporter, Object)
	}

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

	const FileId& AssetImporter::GetMainObject()
	{
		return m_MainObject;
	}

	const Dictionary<FileId, ObjectId>& AssetImporter::GetAssetObjects()
	{
		return m_AssetObjects;
	}

	const bool AssetImporter::IsImported()
	{
		if (m_MainObject > 0)
		{
			Object* mainObject = ObjectDB::GetObjectFromGuid(m_Guid, m_MainObject);
			return mainObject->GetState() != ObjectState::AwaitingLoading;
		}
		return false;
	}

	const bool& AssetImporter::IsRequiringSave()
	{
		return m_RequireSave;
	}

	void AssetImporter::ResetImport()
	{
		if (m_MainObject > 0)
		{
			Object* mainObject = ObjectDB::GetObjectFromGuid(m_Guid, m_MainObject);
			mainObject->SetState(ObjectState::AwaitingLoading);
		}
	}

	void AssetImporter::ImportDataIfNeeded()
	{
		if (GetState() == ObjectState::Loading)
		{
			return;
		}
		if (GetState() == ObjectState::AwaitingLoading)
		{
			AssetImporter::LoadFromMeta(this);
			SetState(ObjectState::Default);
		}
		if (!IsImported())
		{
			ImportData();
		}
	}

	void AssetImporter::Save()
	{
		YamlMetaSerializer serializer;
		serializer.SetGuid(m_Guid);
		serializer.AddObject(this);
		serializer.Serialize(GetMetaFilePath());
		m_RequireSave = false;
	}

	AssetImporter* AssetImporter::CreateNew(const std::size_t& type, const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath)
	{
		auto info = ClassDB::GetInfo(type);
		AssetImporter* importer = static_cast<AssetImporter*>(info.createInstance());
		importer->m_Guid = Guid::Create();
		importer->m_RelativePath = relativePath.string();
		importer->m_RelativeMetaPath = relativeMetaPath.string();
		importer->m_Name = relativePath.stem().string();
		importer->Save();
		importer->m_RequireSave = true; // Importer may get some important data from the asset and should be saved second time
		return importer;
	}

	AssetImporter* AssetImporter::CreateFromMeta(const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath)
	{
		YamlMetaSerializer serializer;
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(relativeMetaPath.string());
		serializer.Deserialize(dataPath.string());

		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			AssetImporter* importer = static_cast<AssetImporter*>(deserializedObjects[0].first);
			Guid guid = serializer.GetGuid();
			importer->m_Guid = guid;
			importer->m_RelativePath = relativePath.string();
			importer->m_RelativeMetaPath = relativeMetaPath.string();
			importer->m_Name = relativePath.stem().string();
			importer->SetState(ObjectState::AwaitingLoading);
			// Data will be imported when it's needed
			//importer->ImportData();
			return importer;
		}
		return nullptr;
	}

	void AssetImporter::LoadFromMeta(AssetImporter* importer)
	{
		YamlMetaSerializer serializer;
		serializer.AddObject(importer);
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(importer->GetRelativeMetaFilePath());
		serializer.Deserialize(dataPath.string());
	}

	void AssetImporter::AddAssetObject(Object* object, const FileId& fileId)
	{
		m_AssetObjects.insert_or_assign(fileId, object->GetObjectId());
	}

	void AssetImporter::SetMainObject(const FileId& id)
	{
		m_MainObject = id;
	}
}
