#include "AssetDB.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Serialization\EditorSerializer.h"
#include "Editor\Assets\Importers\DefaultImporter.h"
#include "Editor\Assets\Importers\NativeAssetImporter.h"
#include "Editor\Assets\ImporterInfoCache.h"
#include "Editor\Assets\DependencyCache.h"
#include "Editor\Misc\PlatformHelper.h"
#include "Editor\Misc\PathHelper.h"

namespace Blueberry
{
	// TODO import data always if it was found in project but not in cache instead of importing on mouse over icon

	Dictionary<String, TypeId> AssetDB::s_ImporterTypes = {};
	List<std::pair<TypeId, ObjectFinalizer*>> AssetDB::s_Finalizers = {};
	List<ObjectId> AssetDB::s_DirtyAssets = {};
	AssetDBRefreshEvent AssetDB::s_AssetDBRefreshed = {};
	FileWatch* AssetDB::s_AssetsWatch = nullptr;

	Dictionary<String, ObjectId> AssetDB::s_RelativePathToImporterId;
	Dictionary<Guid, ObjectId> AssetDB::s_GuidToImporterId;
	Dictionary<Guid, String> AssetDB::s_GuidToRelativePath;

	void AssetDB::Initialize()
	{
		DependencyCache::Load();
		ImporterInfoCache::Load();
		InitializeImporters();
		s_AssetsWatch = FileWatch::Create(StringHelper::ToString(Path::GetAssetsPath()));
	}

	void AssetDB::Shutdown()
	{
		DependencyCache::Save();
		ImporterInfoCache::Save();
		delete s_AssetsWatch;
	}

	void AssetDB::Refresh()
	{
		HashSet<String> pathsToCheck = {};
		for (auto& operation : s_AssetsWatch->GetFileOperations())
		{
			std::filesystem::path path = Path::GetAssetPath(operation.path);
			if (std::filesystem::exists(path))
			{
				if (std::filesystem::is_directory(path))
				{
					for (auto& entry : std::filesystem::recursive_directory_iterator(path))
					{
						String pathString = StringHelper::ToString(entry.path());
						if (StringHelper::EndsWith(pathString, ".meta"))
						{
							pathString = pathString.substr(0, pathString.size() - 5);
						}
						pathsToCheck.insert(pathString);
					}
				}
				else
				{
					String pathString = StringHelper::ToString(path);
					if (StringHelper::EndsWith(pathString, ".meta"))
					{
						pathString = pathString.substr(0, pathString.size() - 5);
					}
					pathsToCheck.insert(pathString);
				}
			}
		}
		if (pathsToCheck.size() > 0)
		{
			for (const String& path : pathsToCheck)
			{
				Guid guid = PathHelper::GetMetaGuid(path + ".meta");
				String relativePath = StringHelper::ToString(std::filesystem::relative(path, Path::GetAssetsPath()));
				auto it = s_GuidToRelativePath.find(guid);
				if (it != s_GuidToRelativePath.end())
				{
					ObjectId importerId = s_GuidToImporterId[guid];
					AssetImporter* importer = static_cast<AssetImporter*>(ObjectDB::GetObject(importerId));

					if (relativePath != it->second)
					{
						importer->Rename(relativePath);
						s_RelativePathToImporterId.erase(it->second);
						s_RelativePathToImporterId.insert_or_assign(relativePath, importerId);
						s_GuidToRelativePath.insert_or_assign(guid, relativePath);
					}

					if (NeedsReimport(importer))
					{
						Guid guid = importer->GetGuid();
						if (HasAssetWithGuidInData(guid))
						{
							DeleteAssetFromData(guid);
							importer->ResetImport();
						}
						OnImportAsset(importer);
						ImporterInfoCache::Save();
					}
				}
				else
				{
					AssetImporter* importer = CreateOrGetImporter(path);
					if (importer != nullptr)
					{
						ObjectId importerId = importer->GetObjectId();
						s_RelativePathToImporterId.insert_or_assign(relativePath, importerId);
						s_GuidToImporterId.insert_or_assign(guid, importerId);
						s_GuidToRelativePath.insert_or_assign(guid, relativePath);
						importer->Rename(relativePath);
						OnImportAsset(importer);
						ImporterInfoCache::Save();
					}
				}
			}
			List<AssetImporter*> deletedImporters;
			for (auto& pair : s_RelativePathToImporterId)
			{
				std::filesystem::path path = Path::GetAssetPath(pair.first);
				if (!std::filesystem::exists(path))
				{
					deletedImporters.push_back(static_cast<AssetImporter*>(ObjectDB::GetObject(pair.second)));
				}
			}
			for (AssetImporter* importer : deletedImporters)
			{
				OnDeleteAsset(importer);
			}
		}
		s_AssetsWatch->ClearFileOperations();
		s_AssetDBRefreshed.Invoke();
	}

	AssetImporter* AssetDB::GetImporter(const String& relativePath)
	{
		if (relativePath.length() == 0)
		{
			return nullptr;
		}
		auto it = s_RelativePathToImporterId.find(relativePath);
		if (it != s_RelativePathToImporterId.end())
		{
			return static_cast<AssetImporter*>(ObjectDB::GetObject(it->second));
		}
		std::filesystem::path path = Path::GetAssetsPath();
		path.append(relativePath);
		return CreateImporter(path, relativePath);
	}

	AssetImporter* AssetDB::GetImporter(const Guid& guid)
	{
		auto it = s_GuidToImporterId.find(guid);
		if (it != s_GuidToImporterId.end())
		{
			return static_cast<AssetImporter*>(ObjectDB::GetObject(it->second));
		}
		return nullptr;
	}

	List<std::pair<Object*, FileId>> AssetDB::LoadAssetObjects(const Guid& guid, const Dictionary<FileId, ObjectId>& existingObjects)
	{
		List<std::pair<Object*, FileId>> objects;
		String assetPath = Path::GetAssetCachePath(guid.ToString());
		EditorSerializer serializer = {};
		serializer.SetGuid(guid);
		for (auto& pair : existingObjects)
		{
			Object* existingObject = ObjectDB::GetObject(pair.second);
			if (existingObject != nullptr)
			{
				serializer.AddObject(existingObject);
			}
		}
		serializer.Deserialize(assetPath, SerializationFlags::HasHeaders);
		serializer.FinalizeObjects();
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
		auto it = s_GuidToRelativePath.find(guid);
		if (it != s_GuidToRelativePath.end())
		{
			return it->second;
		}
		return "";
	}

	String AssetDB::GetAssetCachedDataPath(Object* object)
	{
		const std::filesystem::path& directoryPath = Path::GetAssetCachePath();
		if (!std::filesystem::exists(directoryPath))
		{
			std::filesystem::create_directories(directoryPath);
		}
		return Path::GetAssetCachePath(ObjectDB::GetGuidFromObject(object).ToString());
	}

	void AssetDB::GetDependent(const Guid& guid, HashSet<Guid>& dependent)
	{
		DependencyCache::Get(guid, dependent);
	}

	void AssetDB::SetDependencies(const Guid& guid, const HashSet<Guid>& dependencies)
	{
		DependencyCache::Set(guid, dependencies);
	}

	bool AssetDB::HasAssetWithGuidInData(const Guid& guid)
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		dataPath.append(guid.ToString());
		return std::filesystem::exists(dataPath);
	}

	void AssetDB::CreateAsset(Object* object, const String& relativePath)
	{
		EditorSerializer serializer = {};
		String assetPath = Path::GetAssetsPath(relativePath);
		serializer.SetGuid(ObjectDB::GetGuidFromObject(object));
		serializer.AddObject(object);
		serializer.Serialize(assetPath, SerializationFlags::EditorOnly | (ClassDB::GetInfo(object->GetType())->preferBinary ? SerializationFlags::None : (SerializationFlags::Text | SerializationFlags::HasHeaders)));
		
		AssetImporter* importer = CreateOrGetImporter(assetPath);
		const Guid& guid = importer->GetGuid();
		ObjectDB::AllocateIdToGuid(object, guid, ObjectDB::GetFileIdFromObject(object));
		//ImporterInfoCache::Set(importer);
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
			serializer.Serialize(GetAssetCachedDataPath(objects[0]), SerializationFlags::EditorOnly);
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

	void AssetDB::LoadAsset(const Guid& guid)
	{
		AssetImporter* importer = GetImporter(guid);
		if (importer != nullptr)
		{
			importer->ImportDataIfNeeded();
		}
	}

	void AssetDB::ImportAsset(const String& relativePath)
	{
		String assetPath = Path::GetAssetsPath(relativePath);
		AssetImporter* importer = CreateOrGetImporter(assetPath);
		Guid guid = importer->GetGuid();
		if (HasAssetWithGuidInData(guid))
		{
			DeleteAssetFromData(guid);
			importer->ResetImport();
		}
		OnImportAsset(importer);
	}

	void AssetDB::RenameAsset(const String& fromRelativePath, const String& toRelativePath)
	{
		String fromPath = Path::GetAssetsPath(fromRelativePath);
		String toPath = Path::GetAssetsPath(toRelativePath);
		std::filesystem::rename(fromPath, toPath);
		std::filesystem::rename(fromPath + ".meta", toPath + ".meta");
	}

	void AssetDB::DeleteAsset(const String& relativePath)
	{
		auto it = s_RelativePathToImporterId.find(relativePath);
		if (it != s_RelativePathToImporterId.end())
		{
			String assetPath = Path::GetAssetsPath(relativePath);
			PlatformHelper::MoveToRecycleBin(assetPath);
			PlatformHelper::MoveToRecycleBin(assetPath + ".meta");
			AssetImporter* importer = static_cast<AssetImporter*>(ObjectDB::GetObject(it->second));
			DeleteAssetFromData(importer->GetGuid());
		}
	}

	void AssetDB::DeleteAssetFromData(const Guid& guid)
	{
		String assetPath = Path::GetAssetCachePath(guid.ToString());
		std::filesystem::remove(assetPath);
	}

	void AssetDB::SaveAssets()
	{
		for (ObjectId objectId : s_DirtyAssets)
		{
			Object* object = ObjectDB::GetObject(objectId);
			if (object != nullptr)
			{
				auto pair = ObjectDB::GetGuidAndFileIdFromObject(object);
				auto it = s_GuidToRelativePath.find(pair.first);
				if (it != s_GuidToRelativePath.end())
				{
					String relativePath = s_GuidToRelativePath[pair.first];
					EditorSerializer serializer = {};
					String assetPath = Path::GetAssetsPath(relativePath);
					serializer.SetGuid(ObjectDB::GetGuidFromObject(object));
					serializer.AddObject(object);
					serializer.Serialize(assetPath, SerializationFlags::EditorOnly | (ClassDB::GetInfo(object->GetType())->preferBinary ? SerializationFlags::None : (SerializationFlags::Text | SerializationFlags::HasHeaders)));
				}
				else if (object->IsClassType(AssetImporter::Type))
				{
					AssetImporter* importer = static_cast<AssetImporter*>(object);
					importer->Save();
					importer->ResetImport();
				}
			}
		}
		s_DirtyAssets.clear();
		Refresh();
	}

	AssetDBRefreshEvent& AssetDB::GetAssetDBRefreshed()
	{
		return s_AssetDBRefreshed;
	}

	void AssetDB::InitializeImporters()
	{
		const std::filesystem::path& assetsPath = Path::GetAssetsPath();
		List<AssetImporter*> importersToImport = {};

		auto& end = std::filesystem::recursive_directory_iterator();
		for (auto it = std::filesystem::recursive_directory_iterator(assetsPath); it != end; ++it)
		{
			std::filesystem::path path = it->path();
			if (path.extension() == ".meta")
			{
				continue;
			}

			AssetImporter* importer = CreateOrGetImporter(path);
			if (importer != nullptr)
			{
				if (NeedsReimport(importer))
				{
					Guid guid = importer->GetGuid();
					if (HasAssetWithGuidInData(guid))
					{
						DeleteAssetFromData(guid);
						importer->ResetImport();
					}
					importersToImport.push_back(importer);
				}
			}
		}
		for (AssetImporter* importer : importersToImport)
		{
			OnImportAsset(importer);
			ImporterInfoCache::Save();
		}
	}

	bool AssetDB::NeedsReimport(AssetImporter* importer)
	{
		if (!std::filesystem::exists(importer->GetFilePath()))
		{
			return false;
		}
		if (!ImporterInfoCache::Has(importer))
		{
			return true;
		}
		bool result = importer->GetLastWrite() < std::max(PathHelper::GetLastWriteTime(importer->GetFilePath()), PathHelper::GetLastWriteTime(importer->GetMetaFilePath()));
		return result;
	}

	void AssetDB::OnImportAsset(AssetImporter* importer)
	{
		importer->ImportDataIfNeeded();
		if (importer->IsRequiringSave())
		{
			importer->Save();
		}
		ImporterInfoCache::Set(importer);
	}

	void AssetDB::OnRenameAsset(AssetImporter* importer, const String& toRelativePath)
	{
		const String& fromRelativePath = importer->GetRelativeFilePath();
		importer->Rename(toRelativePath);
		s_RelativePathToImporterId.erase(fromRelativePath);
		s_RelativePathToImporterId.insert_or_assign(toRelativePath, importer->GetObjectId());
		s_GuidToRelativePath.insert_or_assign(importer->GetGuid(), toRelativePath);
	}

	void AssetDB::OnDeleteAsset(AssetImporter* importer)
	{
		const String& relativePath = importer->GetRelativeFilePath();
		const Guid& guid = importer->GetGuid();
		auto snapshot = ObjectDB::GetObjectsFromGuid(guid);
		for (auto& pair : snapshot)
		{
			Object* object = ObjectDB::GetObject(pair.second);
			if (object != nullptr)
			{
				Object::Destroy(object);
			}
		}
		ImporterInfoCache::Clear(importer);
		Object::Destroy(importer);
		s_GuidToRelativePath.erase(guid);
		s_GuidToImporterId.erase(guid);
		s_RelativePathToImporterId.erase(importer->GetRelativeFilePath());
	}
	
	AssetImporter* AssetDB::CreateOrGetImporter(const std::filesystem::path& path)
	{
		// Skip not existing pathes
		if (!std::filesystem::exists(path))
		{
			return nullptr;
		}

		std::filesystem::path extension = path.extension();

		// Skip meta files
		if (extension == ".meta")
		{
			return nullptr;
		}

		std::filesystem::path relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		String relativePathString = StringHelper::ToString(relativePath);
		if (relativePathString.length() == 0)
		{
			return nullptr;
		}
		AssetImporter* existingImporter = GetImporter(relativePathString);
		if (existingImporter != nullptr)
		{
			return existingImporter;
		}

		return CreateImporter(path, relativePath);
	}

	AssetImporter* AssetDB::CreateImporter(const std::filesystem::path& path, const std::filesystem::path& relativePath)
	{
		std::filesystem::path extension = std::filesystem::is_directory(path) ? "" : path.extension();
		String extensionString = StringHelper::ToString(extension);

		std::filesystem::path metaPath = path;
		metaPath += ".meta";

		AssetImporter* importer;
		if (!std::filesystem::exists(metaPath))
		{
			// Create new meta file
			auto importerTypeIt = s_ImporterTypes.find(extensionString);
			if (importerTypeIt != s_ImporterTypes.end())
			{
				importer = AssetImporter::CreateNew(importerTypeIt->second, relativePath);
			}
			else
			{
				importer = AssetImporter::CreateNew(DefaultImporter::Type, relativePath);
				BB_INFO("AssetImporter for extension " << extensionString << " does not exist and default importer was created.");
			}
		}
		else
		{
			// Create importer from meta file
			importer = AssetImporter::CreateFromMeta(relativePath);
			ImporterInfoCache::Get(importer);
		}
		String relativePathString = StringHelper::ToString(relativePath);
		Guid guid = importer->GetGuid();
		ObjectId objectId = importer->GetObjectId();
		s_GuidToRelativePath.insert_or_assign(guid, relativePathString);
		s_GuidToImporterId.insert_or_assign(guid, objectId);
		s_RelativePathToImporterId.insert_or_assign(relativePathString, importer->GetObjectId());
		return importer;
	}

	void AssetDB::Register(const String& extension, const TypeId& importerType)
	{
		s_ImporterTypes.insert_or_assign(extension, importerType);
	}
}
