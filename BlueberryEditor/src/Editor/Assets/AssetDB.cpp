#include "bbpch.h"
#include "AssetDB.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Assets\Importers\DefaultImporter.h"
#include "Editor\Assets\Importers\NativeAssetImporter.h"
#include "rapidyaml\ryml.h"
#include <fstream>
#include <sstream>

namespace Blueberry
{
	std::map<std::string, std::size_t> AssetDB::s_ImporterTypes = std::map<std::string, std::size_t>();
	std::map<std::string, AssetImporter*> AssetDB::s_Importers = std::map<std::string, AssetImporter*>();
	
	std::map<Guid, std::string> AssetDB::s_GuidToPath = std::map<Guid, std::string>();
	std::map<std::string, long long> AssetDB::s_PathModifyCache = std::map<std::string, long long>();
	std::vector<ObjectId> AssetDB::s_DirtyAssets = std::vector<ObjectId>();

	void AssetDB::Refresh()
	{
		LoadModifyCache();
		for (auto& it : std::filesystem::recursive_directory_iterator(Path::GetAssetsPath()))
		{
			AssetImporter* importer = CreateImporter(it.path());
			if (importer != nullptr)
			{
				s_GuidToPath.insert_or_assign(importer->GetGuid(), importer->GetRelativeFilePath());
				// Delete asset from cache if it dirty
				auto lastWriteTime = std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(importer->GetFilePath()).time_since_epoch()).count();
				auto lastWriteIt = s_PathModifyCache.find(importer->GetRelativeFilePath());
				bool needsClearing = false;

				if (lastWriteIt != s_PathModifyCache.end())
				{
					if (lastWriteIt->second < lastWriteTime)
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
				}
				s_PathModifyCache.insert_or_assign(importer->GetRelativeFilePath(), lastWriteTime);
				if (needsClearing)
				{
					SaveModifyCache();
				}
			}
		}
	}

	AssetImporter* AssetDB::Import(const std::string& relativePath)
	{
		auto& dataIt = s_Importers.find(relativePath);
		if (dataIt != s_Importers.end())
		{
			dataIt->second->ImportDataIfNeeded();
			return dataIt->second;
		}
		return nullptr;
	}

	AssetImporter* AssetDB::Import(const Guid& guid)
	{
		auto pathIt = s_GuidToPath.find(guid);
		if (pathIt != s_GuidToPath.end())
		{
			return Import(pathIt->second);
		}
		return nullptr;
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

	std::vector<Object*> AssetDB::LoadAssetObjects(const Guid & guid)
	{
		YamlSerializer serializer;
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		serializer.Deserialize(dataPath.append(guid.ToString().append(".yaml")).string());
		auto& deserializedObjects = serializer.GetDeserializedObjects();
		std::vector<Object*> objects(deserializedObjects.size());
		if (deserializedObjects.size() > 0)
		{
			int i = 0;
			for (auto& pair : deserializedObjects)
			{
				Object* object = pair.first;
				ObjectDB::AllocateIdToGuid(object, guid, pair.second);
				objects[i] = object;
				++i;
			}
		}
		return objects;
	}

	std::string AssetDB::GetAssetCachedDataPath(Object* object, const char* extension)
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		return dataPath.append(ObjectDB::GetGuidFromObject(object).ToString().append(extension)).string();
	}

	bool AssetDB::HasAssetWithGuidInData(const Guid& guid)
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append(guid.ToString().append(".yaml"));
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
		// TODO binary serializer
		YamlSerializer serializer;
		for (Object* object : objects)
		{
			serializer.AddObject(object);
		}
		serializer.Serialize(GetAssetCachedDataPath(objects[0], ".yaml"));
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
		dataPath.append(guid.ToString().append(".yaml"));
		std::filesystem::remove(dataPath);
	}

	void AssetDB::SaveAssets()
	{
		for (auto& objectId : s_DirtyAssets)
		{
			ObjectItem* item = ObjectDB::IdToObjectItem(objectId);
			if (item != nullptr && item->object != nullptr)
			{
				auto pair = ObjectDB::GetGuidAndFileIdFromObject(item->object);
				std::string relativePath = s_GuidToPath[pair.first];
				YamlSerializer serializer;
				auto dataPath = Path::GetAssetsPath();
				dataPath.append(relativePath);
				serializer.AddObject(item->object);
				serializer.Serialize(dataPath.string());
				BB_INFO("Asset was saved to the path: " << relativePath);
			}
		}
		s_DirtyAssets.clear();
	}

	AssetImporter* AssetDB::CreateImporter(const std::filesystem::path& path)
	{
		// Skip directories and not existing pathes
		if (!std::filesystem::exists(path) || std::filesystem::is_directory(path))
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

		auto metaPath = path;
		metaPath += ".meta";

		AssetImporter* importer;
		if (!std::filesystem::exists(metaPath))
		{
			// Create new meta file
			auto importerTypeIt = s_ImporterTypes.find(extensionString);
			if (importerTypeIt != s_ImporterTypes.end())
			{
				importer = AssetImporter::Create(importerTypeIt->second, relativePath, relativeMetaPath);
			}
			else
			{
				importer = AssetImporter::Create(DefaultImporter::Type, relativePath, relativeMetaPath);
				BB_INFO("AssetImporter for extension " << extensionString << " does not exist and default importer was created.");
			}
		}
		else
		{
			// Create importer from meta file
			importer = AssetImporter::Load(relativePath, relativeMetaPath);
		}
		s_Importers.insert_or_assign(relativePathString, importer);
		return importer;
	}

	void AssetDB::LoadModifyCache()
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append("PathModifyCache.yaml");

		if (std::filesystem::exists(dataPath))
		{
			std::ifstream input;
			input.open(dataPath, std::ifstream::in);
			std::string line;
			while (std::getline(input, line))
			{
				int tabIndex = line.find('\t');
				std::string path = line.substr(0, tabIndex);
				long long time = std::stoll(line.substr(tabIndex + 1));
				s_PathModifyCache.insert_or_assign(path, time);
			}
			input.close();
		}
	}

	void AssetDB::SaveModifyCache()
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append("PathModifyCache.yaml");

		std::ofstream output;
		output.open(dataPath, std::ofstream::out);
		std::stringstream sstream;
		for (auto& pair : s_PathModifyCache)
		{
			std::string path = pair.first;
			long long time = pair.second;
			sstream << path << "\t" << time << std::endl;
		}
		output << sstream.rdbuf();
		output.close();
	}

	void AssetDB::Register(const std::string& extension, const std::size_t& importerType)
	{
		s_ImporterTypes.insert_or_assign(extension, importerType);
	}
}
