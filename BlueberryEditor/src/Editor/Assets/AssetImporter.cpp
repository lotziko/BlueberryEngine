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

	const std::unordered_map<FileId, ObjectId>& AssetImporter::GetImportedObjects()
	{
		return m_ImportedObjects;
	}

	const FileId& AssetImporter::GetMainObject()
	{
		return m_MainObject;
	}

	const bool& AssetImporter::IsImported()
	{
		if (m_ImportedObjects.size() > 0)
		{
			Object* firstObject = ObjectDB::GetObject(m_ImportedObjects.begin()->second);
			return firstObject->GetState() != ObjectState::AwaitingLoading;
		}
		return false;
	}

	const Texture2D* AssetImporter::GetIcon()
	{
		return m_Icon;
	}

	void AssetImporter::ResetImport()
	{
		for (auto& pair : m_ImportedObjects)
		{
			Object* object = ObjectDB::GetObject(pair.second);
			object->SetState(ObjectState::AwaitingLoading);
		}
	}

	void AssetImporter::ImportDataIfNeeded()
	{
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
	}

	AssetImporter* AssetImporter::Create(const size_t& type, const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath)
	{
		auto info = ClassDB::GetInfo(type);
		AssetImporter* importer = (AssetImporter*)info.createInstance();
		importer->m_Guid = Guid::Create();
		importer->m_RelativePath = relativePath.string();
		importer->m_RelativeMetaPath = relativeMetaPath.string();
		importer->m_Name = relativePath.stem().string();
		importer->m_Icon = (Texture2D*)AssetLoader::Load(importer->GetIconPath());
		importer->Save();
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
			AssetImporter* importer = (AssetImporter*)deserializedObjects[0].first;
			Guid guid = serializer.GetGuid();
			importer->m_Guid = guid;
			importer->m_RelativePath = relativePath.string();
			importer->m_RelativeMetaPath = relativeMetaPath.string();
			importer->m_Name = relativePath.stem().string();
			importer->m_Icon = (Texture2D*)AssetLoader::Load(importer->GetIconPath());
			// Data will be imported when it's needed
			//importer->ImportData();
			return importer;
		}
		return nullptr;
	}

	void AssetImporter::BindProperties()
	{
	}

	void AssetImporter::AddImportedObject(Object* object, const FileId& fileId)
	{
		m_ImportedObjects.insert_or_assign(fileId, object->GetObjectId());
	}

	void AssetImporter::SetMainObject(const FileId& id)
	{
		m_MainObject = id;
	}

	std::string AssetImporter::GetIconPath()
	{
		return "assets/icons/FileIcon.png";
	}
}
