#include "bbpch.h"
#include "AssetDB.h"

#include "Editor\Serialization\AssetImporter.h"
#include "rapidyaml\ryml.h"

#include <fstream>

namespace Blueberry
{
	std::map<std::string, long long> AssetDB::s_PathModifyCache = std::map<std::string, long long>();
	std::map<std::string, std::size_t> AssetDB::s_ImporterTypes = std::map<std::string, std::size_t>();
	std::map<Guid, Ref<AssetImporter>> AssetDB::s_Importers = std::map<Guid, Ref<AssetImporter>>();
	std::map<Guid, Ref<Object>> AssetDB::s_ImportedObjects = std::map<Guid, Ref<Object>>();

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

	bool AssetDB::HasAssetWithGuidInData(const Guid& guid)
	{
		auto dataPath = Path::GetDataPath();
		dataPath.append(guid.ToString().append(".yaml"));
		return std::filesystem::exists(dataPath);
	}

	void AssetDB::SaveAssetObject(Ref<Object> object)
	{
		auto dataPath = Path::GetDataPath();

		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}

		dataPath.append(object->GetGuid().ToString().append(".yaml"));

		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;
		object->Serialize(root);
		YamlHelper::Save(tree, dataPath.string());
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

		auto pathString = path.string();
		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		auto relativePathString = relativePath.string();
		
		auto lastWriteTime = std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::last_write_time(path).time_since_epoch()).count();

		auto metaPath = path;
		metaPath += ".meta";
		auto metaPathString = metaPath.string();

		Ref<AssetImporter> importer;
		if (!std::filesystem::exists(metaPath))
		{
			// Create new meta file
			auto importerTypeIt = s_ImporterTypes.find(extensionString);
			if (importerTypeIt != s_ImporterTypes.end())
			{
				importer = AssetImporter::Create(importerTypeIt->second, pathString, metaPathString);
			}
			else
			{
				BB_ERROR(std::string() << "AssetImporter for extension " << extensionString << " does not exist.");
				return;
			}
		}
		else
		{
			// Create importer from meta file
			importer = AssetImporter::Load(pathString, metaPathString);
		}
		s_Importers.insert({ importer->GetGuid(), importer });
	}

	void AssetDB::Register(const std::string& extension, const std::size_t& importerType)
	{
		s_ImporterTypes.insert_or_assign(extension, importerType);
	}
}
