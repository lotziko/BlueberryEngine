#include "AssetDB.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Tools\StringConverter.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Serialization\EditorSerializer.h"
#include "Editor\Assets\Importers\DefaultImporter.h"
#include "Editor\Assets\Importers\NativeAssetImporter.h"
#include "Editor\Assets\PathModifyCache.h"
#include "Editor\Assets\ImporterInfoCache.h"
#include "Editor\Assets\DependencyCache.h"
#include "Editor\Misc\PathHelper.h"
#include "Editor\Misc\PlatformHelper.h"

namespace Blueberry
{
	Dictionary<String, TypeId> AssetDB::s_ImporterTypes = {};
	Dictionary<String, AssetImporter*> AssetDB::s_Importers = {};
	List<std::pair<TypeId, ObjectFinalizer*>> AssetDB::s_Finalizers = {};
	
	Dictionary<Guid, String> AssetDB::s_GuidToPath = {};
	List<ObjectId> AssetDB::s_DirtyAssets = {};

	AssetDBRefreshEvent AssetDB::s_AssetDBRefreshed = {};

	// TODO import data always if it was found in project but not in cache instead of importing on mouse over icon
	void AssetDB::Refresh()
	{
		List<AssetImporter*> importersToImport = {};
		DependencyCache::Load();
		PathModifyCache::Load();
		ImporterInfoCache::Load();
		
		for (auto& it : std::filesystem::recursive_directory_iterator(Path::GetAssetsPath()))
		{
			AssetImporter* importer = CreateOrGetImporter(it.path());
			if (importer != nullptr)
			{
				s_GuidToPath.insert_or_assign(importer->GetGuid(), importer->GetRelativeFilePath());
				// Delete asset from cache if it is dirty
				auto assetLastWriteTime = PathHelper::GetLastWriteTime(importer->GetFilePath());
				auto metaLastWriteTime = PathHelper::GetLastWriteTime(importer->GetMetaFilePath());
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
						importer->ResetImport();
					}
					importersToImport.push_back(importer);
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
			try
			{
				importer->ImportDataIfNeeded();
			}
			catch (...)
			{
				BB_ERROR("Failed to import asset.");
				break;
			}
			ImporterInfoCache::Set(importer);
			if (importer->IsRequiringSave())
			{
				importer->Save();
			}
		}
		ImporterInfoCache::Save();
		s_AssetDBRefreshed.Invoke();
	}

	AssetImporter* AssetDB::GetImporter(const String& relativePath)
	{
		auto& dataIt = s_Importers.find(relativePath);
		if (dataIt != s_Importers.end())
		{
			return dataIt->second;
		}
		auto path = Path::GetAssetsPath();
		path.append(relativePath);
		return CreateImporter(path);
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

	List<std::pair<Object*, FileId>> AssetDB::LoadAssetObjects(const Guid& guid, const Dictionary<FileId, ObjectId>& existingObjects)
	{
		List<std::pair<Object*, FileId>> objects;
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		String path = String(dataPath.append(guid.ToString()).string());
		EditorSerializer serializer = {};
		serializer.SetGuid(guid);
		for (auto& object : existingObjects)
		{
			Object* existingObject = ObjectDB::GetObject(object.second);
			if (existingObject != nullptr)
			{
				serializer.AddObject(existingObject);
			}
		}
		serializer.Deserialize(path);
		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			for (auto& pair : deserializedObjects)
			{
				ObjectDB::AllocateIdToGuid(pair.first, guid, pair.second);
				objects.push_back(std::make_pair(ObjectDB::GetObject(pair.first), pair.second));
			}
		}
		return objects;
	}

	const String AssetDB::GetRelativeAssetPath(Object* object)
	{
		// TODO make some cache instead
		if (object == nullptr)
		{
			return "";
		}

		AssetImporter* importer = nullptr;
		if (object->IsClassType(AssetImporter::Type))
		{
			importer = static_cast<AssetImporter*>(object);
		}

		if (ObjectDB::HasGuid(object))
		{
			importer = GetImporter(ObjectDB::GetGuidFromObject(object));
		}

		if (importer != nullptr)
		{
			return importer->GetRelativeFilePath();
		}
		return "";
	}

	const String AssetDB::GetRelativePath(const Guid& guid)
	{
		auto it = s_GuidToPath.find(guid);
		if (it != s_GuidToPath.end())
		{
			return it->second;
		}
		return "";
	}

	String AssetDB::GetAssetCachedDataPath(Object* object)
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		return String(dataPath.append(ObjectDB::GetGuidFromObject(object).ToString()).string());
	}

	void AssetDB::GetDependent(const Guid& guid, HashSet<Guid>& dependent)
	{
		DependencyCache::Get(guid, dependent);
	}

	void AssetDB::SetDependencies(const Guid& guid, const HashSet<Guid>& dependencies)
	{
		DependencyCache::Set(guid, dependencies);
		DependencyCache::Save();
	}

	bool AssetDB::HasAssetWithGuidInData(const Guid& guid)
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append(guid.ToString());
		return std::filesystem::exists(dataPath);
	}

	void AssetDB::CreateAsset(Object* object, const String& relativePath)
	{
		EditorSerializer serializer;
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(relativePath);
		serializer.SetGuid(ObjectDB::GetGuidFromObject(object));
		serializer.AddObject(object);
		serializer.Serialize(String(dataPath.string()), !ClassDB::GetInfo(object->GetType())->preferBinary);
		BB_INFO("Asset was saved to the path: " << relativePath);

		AssetImporter* importer = CreateOrGetImporter(dataPath);
		ObjectDB::AllocateIdToGuid(object, importer->GetGuid(), ObjectDB::GetFileIdFromObject(object));
	}

	void AssetDB::SaveAssetObjectsToCache(const List<Object*>& objects)
	{
		if (objects.size() > 0)
		{
			EditorSerializer serializer = {};
			serializer.SetGuid(ObjectDB::GetGuidFromObject(objects[0]));
			for (Object* object : objects)
			{
				serializer.AddObject(object);
			}
			serializer.Serialize(GetAssetCachedDataPath(objects[0]), false);
		}
	}

	void AssetDB::SetDirty(Object* object)
	{
		if (std::find(s_DirtyAssets.begin(), s_DirtyAssets.end(), object->GetObjectId()) != s_DirtyAssets.end())
		{
			return;
		}

		s_DirtyAssets.push_back(object->GetObjectId());
	}

	void AssetDB::DeleteAsset(Object* object)
	{
		Guid guid;
		AssetImporter* importer;
		if (object->IsClassType(AssetImporter::Type))
		{
			importer = static_cast<AssetImporter*>(object);
			guid = importer->GetGuid();
		}
		else
		{
			guid = ObjectDB::GetGuidFromObject(object);
			auto it = s_GuidToPath.find(guid);
			if (it != s_GuidToPath.end())
			{
				importer = s_Importers[it->second];
			}
			else
			{
				return;
			}
		}
		
		PlatformHelper::MoveToRecycleBin(StringConverter::StringToWide(importer->GetFilePath()) + L'\0');
		PlatformHelper::MoveToRecycleBin(StringConverter::StringToWide(importer->GetMetaFilePath()) + L'\0');
		FileId mainObject = importer->GetMainObject();
		if (mainObject != 0)
		{
			Object::Destroy(ObjectDB::GetObjectFromGuid(guid, mainObject));
		}
		for (auto& pair : importer->GetAssetObjects())
		{
			Object::Destroy(ObjectDB::GetObject(pair.second));
		}
		Object::Destroy(importer);
		s_GuidToPath.erase(guid);
		s_Importers.erase(importer->GetRelativeFilePath());
		s_AssetDBRefreshed.Invoke();
	}

	void AssetDB::DeleteAssetFromData(const Guid& guid)
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append(guid.ToString());
		std::filesystem::remove(dataPath);
	}

	void AssetDB::MarkForReimport(const Guid& guid)
	{
		auto it = s_GuidToPath.find(guid);
		if (it != s_GuidToPath.end())
		{
			PathModifyCache::Clear(it->second);
			PathModifyCache::Save();
		}
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
					String relativePath = s_GuidToPath[pair.first];
					EditorSerializer serializer;
					auto dataPath = Path::GetAssetsPath();
					dataPath.append(relativePath);
					serializer.SetGuid(ObjectDB::GetGuidFromObject(object));
					serializer.AddObject(object);
					serializer.Serialize(String(dataPath.string()), !ClassDB::GetInfo(object->GetType())->preferBinary);
					BB_INFO("Asset was saved to the path: " << relativePath);
				}
				else if (object->IsClassType(AssetImporter::Type))
				{
					AssetImporter* importer = static_cast<AssetImporter*>(object);
					importer->Save();
					importer->ResetImport();
					// Maybe refresh always after save?
					Refresh();
				}
			}
		}
		s_DirtyAssets.clear();
	}

	AssetDBRefreshEvent& AssetDB::GetAssetDBRefreshed()
	{
		return s_AssetDBRefreshed;
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
		AssetImporter* existingImporter = GetImporter(String(relativePath.string()));
		if (existingImporter != nullptr)
		{
			return existingImporter;
		}

		return CreateImporter(relativePath);
	}

	AssetImporter* AssetDB::CreateImporter(const std::filesystem::path& path)
	{
		auto extension = std::filesystem::is_directory(path) ? "" : path.extension();
		auto extensionString = extension.string();

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		auto relativeMetaPath = relativePath;
		relativeMetaPath += ".meta";
		auto relativeMetaPathString = relativeMetaPath.string();

		auto metaPath = path;
		metaPath += ".meta";

		AssetImporter* importer;
		if (!std::filesystem::exists(metaPath))
		{
			// Create new meta file
			auto importerTypeIt = s_ImporterTypes.find(String(extensionString));
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
		s_Importers.insert_or_assign(String(relativePath.string()), importer);
		return importer;
	}

	void AssetDB::Register(const String& extension, const TypeId& importerType)
	{
		s_ImporterTypes.insert_or_assign(extension, importerType);
	}
}
