#include "bbpch.h"
#include "AssetDB.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Serialization\BinarySerializer.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Assets\Importers\DefaultImporter.h"
#include "Editor\Assets\Importers\NativeAssetImporter.h"
#include "Editor\Assets\PathModifyCache.h"
#include "Editor\Assets\ImporterInfoCache.h"

namespace Blueberry
{
	std::unordered_map<std::string, std::size_t> AssetDB::s_ImporterTypes = std::unordered_map<std::string, std::size_t>();
	std::unordered_map<std::string, AssetImporter*> AssetDB::s_Importers = std::unordered_map<std::string, AssetImporter*>();
	
	std::unordered_map<Guid, std::string> AssetDB::s_GuidToPath = std::unordered_map<Guid, std::string>();
	std::vector<ObjectId> AssetDB::s_DirtyAssets = std::vector<ObjectId>();

	// TODO import data always if it was found in project but not in cache instead of importing on mouse over icon
	void AssetDB::Refresh()
	{
		std::vector<AssetImporter*> importersToImport;
		PathModifyCache::Load();
		ImporterInfoCache::Load();
		
		for (auto& it : std::filesystem::recursive_directory_iterator(Path::GetAssetsPath()))
		{
			AssetImporter* importer = CreateOrGetImporter(it.path());
			if (importer != nullptr)
			{
				s_GuidToPath.insert_or_assign(importer->GetGuid(), importer->GetRelativeFilePath());
				// Delete asset from cache if it is dirty
				auto assetLastWriteTime = std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(importer->GetFilePath()).time_since_epoch()).count();
				auto metaLastWriteTime = std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(importer->GetMetaFilePath()).time_since_epoch()).count();
				auto lastWriteCacheInfo = PathModifyCache::Get(importer->GetRelativeFilePath());
				bool needsClearing = false;

				if (!ImporterInfoCache::Has(importer))
				{
					needsClearing = true;
				}

				if (lastWriteCacheInfo.assetLastWrite > 0 || lastWriteCacheInfo.metaLastWrite > 0)
				{
					if (lastWriteCacheInfo.assetLastWrite < assetLastWriteTime || lastWriteCacheInfo.metaLastWrite < metaLastWriteTime)
					{
						needsClearing = true;
					}
				}
				else
				{
					needsClearing = true;
				}
				if (needsClearing)
				{
					Guid guid = importer->GetGuid();
					if (HasAssetWithGuidInData(guid))
					{
						DeleteAssetFromData(guid);
					}
					importersToImport.emplace_back(importer);
				}
				PathModifyCache::Set(importer->GetRelativeFilePath(), { assetLastWriteTime, metaLastWriteTime });
				if (needsClearing)
				{
					PathModifyCache::Save();
				}
			}
		}
		
		for (AssetImporter* importer : importersToImport)
		{
			importer->ImportDataIfNeeded();
			ImporterInfoCache::Set(importer);
		}
		ImporterInfoCache::Save();
	}

	AssetImporter* AssetDB::GetImporter(const std::string& relativePath)
	{
		auto& dataIt = s_Importers.find(relativePath);
		if (dataIt != s_Importers.end())
		{
			return dataIt->second;
		}
		return nullptr;
	}

	AssetImporter* AssetDB::GetImporter(const Guid& guid)
	{
		auto pathIt = s_GuidToPath.find(guid);
		if (pathIt != s_GuidToPath.end())
		{
			return GetImporter(pathIt->second);
		}
		return nullptr;
	}
	
	std::vector<std::pair<Object*, FileId>> AssetDB::LoadAssetObjects(const Guid& guid, const std::unordered_map<FileId, ObjectId>& existingObjects)
	{
		BinarySerializer/*YamlSerializer*/ serializer;
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		for (auto& object : existingObjects)
		{
			Object* existingObject = ObjectDB::GetObject(object.second);
			if (existingObject != nullptr)
			{
				serializer.AddObject(existingObject);
			}
		}
		serializer.Deserialize(dataPath.append(guid.ToString()).string());
		auto& deserializedObjects = serializer.GetDeserializedObjects();
		std::vector<std::pair<Object*, FileId>> objects(deserializedObjects.size());
		if (deserializedObjects.size() > 0)
		{
			int i = 0;
			for (auto& pair : deserializedObjects)
			{
				objects[i] = std::make_pair(pair.first, pair.second);
				++i;
			}
		}
		return objects;
	}

	std::string AssetDB::GetAssetCachedDataPath(Object* object)
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		return dataPath.append(ObjectDB::GetGuidFromObject(object).ToString()).string();
	}

	bool AssetDB::HasAssetWithGuidInData(const Guid& guid)
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append(guid.ToString());
		return std::filesystem::exists(dataPath);
	}

	void AssetDB::CreateAsset(Object* object, const std::string& relativePath)
	{
		Guid guid = Guid::Create();
		ObjectDB::AllocateIdToGuid(object, guid, 1);

		YamlSerializer serializer;
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(relativePath);
		serializer.AddObject(object);
		serializer.Serialize(dataPath.string());
		BB_INFO("Asset was saved to the path: " << relativePath);
		Refresh();
	}

	void AssetDB::SaveAssetObjectsToCache(const std::vector<Object*>& objects)
	{
		BinarySerializer/*YamlSerializer*/ serializer;
		for (Object* object : objects)
		{
			serializer.AddObject(object);
		}
		serializer.Serialize(GetAssetCachedDataPath(objects[0]));
	}

	void AssetDB::SetDirty(Object* object)
	{
		if (std::find(s_DirtyAssets.begin(), s_DirtyAssets.end(), object->GetObjectId()) != s_DirtyAssets.end())
		{
			return;
		}

		s_DirtyAssets.emplace_back(object->GetObjectId());
	}

	void AssetDB::DeleteAssetFromData(const Guid& guid)
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append(guid.ToString());
		std::filesystem::remove(dataPath);
	}

	void AssetDB::SaveAssets()
	{
		for (auto& objectId : s_DirtyAssets)
		{
			Object* object = ObjectDB::GetObject(objectId);
			if (object != nullptr)
			{
				auto pair = ObjectDB::GetGuidAndFileIdFromObject(object);
				auto it = s_GuidToPath.find(pair.first);
				if (it != s_GuidToPath.end())
				{
					std::string relativePath = s_GuidToPath[pair.first];
					YamlSerializer serializer;
					auto dataPath = Path::GetAssetsPath();
					dataPath.append(relativePath);
					serializer.AddObject(object);
					serializer.Serialize(dataPath.string());
					BB_INFO("Asset was saved to the path: " << relativePath);
				}
				else if (object->IsClassType(AssetImporter::Type))
				{
					AssetImporter* importer = (AssetImporter*)object;
					importer->Save();
					importer->ResetImport();
					// Maybe refresh always after save?
					Refresh();
				}
			}
		}
		s_DirtyAssets.clear();
	}

	AssetImporter* AssetDB::CreateOrGetImporter(const std::filesystem::path& path)
	{
		// Skip not existing pathes
		if (!std::filesystem::exists(path))
		{
			return nullptr;
		}

		auto extension = path.extension();
		auto extensionString = extension.string();

		// Skip meta files
		if (extensionString == ".meta")
		{
			return nullptr;
		}

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		auto relativePathString = relativePath.string();
		auto relativeMetaPath = relativePath;
		relativeMetaPath += ".meta";
		auto relativeMetaPathString = relativeMetaPath.string();

		AssetImporter* existingImporter = GetImporter(relativePathString);
		if (existingImporter != nullptr)
		{
			return existingImporter;
		}

		auto metaPath = path;
		metaPath += ".meta";

		AssetImporter* importer;
		if (!std::filesystem::exists(metaPath))
		{
			// Create new meta file
			auto importerTypeIt = s_ImporterTypes.find(extensionString);
			if (importerTypeIt != s_ImporterTypes.end())
			{
				importer = AssetImporter::CreateNew(importerTypeIt->second, relativePath, relativeMetaPath);
			}
			else
			{
				importer = AssetImporter::CreateNew(DefaultImporter::Type, relativePath, relativeMetaPath);
				BB_INFO("AssetImporter for extension " << extensionString << " does not exist and default importer was created.");
			}
		}
		else
		{
			// Create importer from meta file
			importer = AssetImporter::CreateFromMeta(relativePath, relativeMetaPath);
			ImporterInfoCache::Get(importer);
		}
		s_Importers.insert_or_assign(relativePathString, importer);
		return importer;
	}

	void AssetDB::Register(const std::string& extension, const std::size_t& importerType)
	{
		s_ImporterTypes.insert_or_assign(extension, importerType);
	}
}
