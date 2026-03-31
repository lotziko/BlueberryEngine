#include "AssetImporter.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Assets\AssetLoader.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Serialization\MetaSerializer.h"
#include "Editor\Misc\PlatformHelper.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, Object)
	{
		DEFINE_BASE_FIELDS(AssetImporter, Object)
	}

	const Guid& AssetImporter::GetGuid() const
	{
		return m_Guid;
	}

	String AssetImporter::GetFilePath()
	{
		std::filesystem::path dataPath = Path::GetAssetsPath();
		dataPath.append(m_RelativePath);
		return String(dataPath.string());
	}

	String AssetImporter::GetMetaFilePath()
	{
		std::filesystem::path dataPath = Path::GetAssetsPath();
		dataPath.append(m_RelativeMetaPath);
		return String(dataPath.string());
	}

	const String& AssetImporter::GetRelativeFilePath()
	{
		return m_RelativePath;
	}

	const String& AssetImporter::GetRelativeMetaFilePath()
	{
		return m_RelativeMetaPath;
	}

	FileId AssetImporter::GetMainObject() const
	{
		return m_MainObject;
	}

	const Dictionary<FileId, ObjectId>& AssetImporter::GetAssetObjects()
	{
		return m_AssetObjects;
	}

	bool AssetImporter::IsImported()
	{
		if (!IsImportable())
		{
			return true;
		}
		if (m_MainObject > 0)
		{
			Object* mainObject = ObjectDB::GetObjectFromGuid(m_Guid, m_MainObject);
			if (mainObject != nullptr)
			{
				return mainObject->GetState() != ObjectState::AwaitingLoading;
			}
		}
		return false;
	}

	bool AssetImporter::IsRequiringReimport() const
	{
		Guid guid = GetGuid();
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			return false;
		}
		return true;
	}

	bool AssetImporter::IsRequiringSave() const
	{
		return m_RequireSave;
	}

	void AssetImporter::ResetImport()
	{
		if (m_MainObject > 0)
		{
			Object* mainObject = ObjectDB::GetObjectFromGuid(m_Guid, m_MainObject);
			if (mainObject != nullptr)
			{
				mainObject->SetState(ObjectState::AwaitingLoading);
			}
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
			if (IsRequiringReimport())
			{
				PlatformHelper::ShowProgressBar("Importing assets", GetRelativeFilePath());
				ImportData();
				PlatformHelper::HideProgressBar();
			}
			else
			{
				PlatformHelper::ShowProgressBar("Loading assets", GetRelativeFilePath());
				LoadData();
				PlatformHelper::HideProgressBar();
			}
		}
	}

	void AssetImporter::Save()
	{
		MetaSerializer serializer = {};
		serializer.SetGuid(m_Guid);
		serializer.AddObject(this);
		serializer.Serialize(GetMetaFilePath(), true);
		m_RequireSave = false;
	}

	void AssetImporter::SaveAndReimport()
	{
		Save();
		ResetImport();
		if (IsRequiringReimport())
		{
			PlatformHelper::ShowProgressBar("Importing assets", GetRelativeFilePath());
			ImportData();
			PlatformHelper::HideProgressBar();
		}
		else
		{
			LoadData();
		}
	}

	void AssetImporter::Rename(const std::filesystem::path& relativePath)
	{
		m_Name = relativePath.stem().string();
		m_RelativePath = relativePath.string();
		m_RelativeMetaPath = relativePath.string() + ".meta";
	}

	long long AssetImporter::GetLastWrite() const
	{
		return m_LastWrite;
	}

	AssetImporter* AssetImporter::CreateNew(const TypeId& type, const std::filesystem::path& relativePath)
	{
		const ClassInfo* info = ClassDB::GetInfo(type);
		AssetImporter* importer = static_cast<AssetImporter*>(info->Create());
		importer->m_Guid = Guid::Create();
		importer->m_RelativePath = relativePath.string();
		importer->m_RelativeMetaPath = relativePath.string() + ".meta";
		importer->m_Name = relativePath.stem().string();
		importer->Save();
		importer->m_RequireSave = true; // Importer may get some important data from the asset and should be saved second time
		return importer;
	}

	AssetImporter* AssetImporter::CreateFromMeta(const std::filesystem::path& relativePath)
	{
		std::filesystem::path relativeMetaPath = relativePath;
		relativeMetaPath += ".meta";

		MetaSerializer serializer = {};
		String metaPath = Path::GetAssetsPath(String(relativeMetaPath.string()));
		serializer.Deserialize(metaPath);

		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			AssetImporter* importer = static_cast<AssetImporter*>(ObjectDB::GetObject(deserializedObjects[0].first));
			Guid guid = serializer.GetGuid();
			importer->m_Guid = guid;
			importer->m_RelativePath = relativePath.string();
			importer->m_RelativeMetaPath = relativePath.string() + ".meta";
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
		MetaSerializer serializer = {};
		serializer.AddObject(importer);
		serializer.Deserialize(importer->GetMetaFilePath());
	}

	bool AssetImporter::IsImportable() const
	{
		return true;
	}

	void AssetImporter::LoadData()
	{
		Guid guid = GetGuid();
		AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
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
