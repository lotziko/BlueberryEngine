#include "bbpch.h"
#include "AssetDB.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Editor\Serialization\AssetImporter.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Editor\Serialization\Concrete\DefaultImporter.h"
#include "rapidyaml\ryml.h"

namespace Blueberry
{
	std::map<std::string, long long> AssetDB::s_PathModifyCache = std::map<std::string, long long>();
	std::map<std::string, std::size_t> AssetDB::s_ImporterTypes = std::map<std::string, std::size_t>();
	std::map<std::string, AssetImporter*> AssetDB::s_Importers = std::map<std::string, AssetImporter*>();

	void AssetDB::ImportAll()
	{
		for (auto& it : std::filesystem::recursive_directory_iterator(Path::GetAssetsPath()))
		{
			Import(it.path());
		}
	}

	void AssetDB::Import(const std::string& path)
	{
		Import(std::filesystem::path(path));
	}

	AssetImporter* AssetDB::GetImporter(const std::string& path)
	{
		auto& importerIt = s_Importers.find(path);
		if (importerIt != s_Importers.end())
		{
			return importerIt->second;
		}
		return nullptr;
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

	void AssetDB::SaveAssetObject(Object* object, const std::string& relativePath)
	{
		YamlSerializer serializer;
		auto dataPath = Path::GetAssetsPath();
		dataPath.append(relativePath);
		serializer.AddObject(object);
		serializer.Serialize(dataPath.string());
	}

	void AssetDB::SaveAssetObjectToCache(Object* object)
	{
		// TODO binary serializer
		YamlSerializer serializer;
		serializer.AddObject(object);
		serializer.Serialize(GetAssetCachedDataPath(object, ".yaml"));
	}

	void AssetDB::Import(const std::filesystem::path& path)
	{
		// Skip directories and not existing pathes
		if (!std::filesystem::exists(path) || std::filesystem::is_directory(path))
		{
			return;
		}

		auto extension = path.extension();
		auto extensionString = extension.string();

		// Skip meta files
		if (extensionString == ".meta")
		{
			return;
		}

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		auto relativePathString = relativePath.string();
		
		auto lastWriteTime = std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(path).time_since_epoch()).count();
		auto lastWriteIt = s_PathModifyCache.find(relativePathString);
		bool needsImport = false;
		
		if (lastWriteIt != s_PathModifyCache.end())
		{
			if (lastWriteIt->second < lastWriteTime)
			{
				needsImport = true;
			}
		}
		else
		{
			needsImport = true;
		}

		if (needsImport)
		{
			auto metaPath = path;
			metaPath += ".meta";

			AssetImporter* importer;
			if (!std::filesystem::exists(metaPath))
			{
				// Create new meta file
				auto importerTypeIt = s_ImporterTypes.find(extensionString);
				if (importerTypeIt != s_ImporterTypes.end())
				{
					importer = AssetImporter::Create(importerTypeIt->second, path, metaPath);
				}
				else
				{
					importer = AssetImporter::Create(DefaultImporter::Type, path, metaPath);
					BB_INFO(std::string() << "AssetImporter for extension " << extensionString << " does not exist and default importer was created.");
				}
			}
			else
			{
				// Create importer from meta file
				importer = AssetImporter::Load(path, metaPath);
			}
			s_Importers.insert_or_assign(metaPath.string(), importer);
			s_PathModifyCache.insert_or_assign(relativePathString, lastWriteTime);
		}
	}

	void AssetDB::Register(const std::string& extension, const std::size_t& importerType)
	{
		s_ImporterTypes.insert_or_assign(extension, importerType);
	}
}
