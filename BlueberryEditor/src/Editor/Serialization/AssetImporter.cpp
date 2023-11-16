#include "bbpch.h"

#include <fstream>
#include "AssetImporter.h"
#include "Blueberry\Core\ClassDB.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, AssetImporter)

	const Guid& AssetImporter::GetGuid()
	{
		return m_Guid;
	}

	const std::string& AssetImporter::GetFilePath()
	{
		return m_Path;
	}

	const std::string& AssetImporter::GetMetaFilePath()
	{
		return m_MetaPath;
	}

	void AssetImporter::Save()
	{
		YAML::Emitter out;
		out << YAML::BeginMap;
		out << YAML::Key << "Guid" << YAML::Value << m_Guid;
		out << YAML::Key << m_Type << YAML::BeginMap;
		SerializeMeta(out);
		out << YAML::EndMap;
		out << YAML::EndMap;
		std::ofstream fout(m_MetaPath);
		fout << out.c_str();
	}

	Ref<AssetImporter> AssetImporter::Create(const size_t& type, const std::string& path, const std::string& metaPath)
	{
		auto info = ClassDB::GetInfo(type);
		Ref<AssetImporter> importer = std::dynamic_pointer_cast<AssetImporter>(info.createInstance());
		importer->m_Guid = Guid::Create();
		importer->m_Type = info.name;
		importer->m_Path = path;
		importer->m_MetaPath = metaPath;
		importer->Save();
		importer->ImportData();
		return importer;
	}

	Ref<AssetImporter> AssetImporter::Load(const std::string& path, const std::string& metaPath)
	{
		YAML::Node node = YAML::LoadFile(metaPath);
		Guid guid = node["Guid"].as<Guid>();
		std::string type;
		for (auto& pair : node)
		{
			std::string key = pair.first.as<std::string>();
			if (key != "Guid")
			{
				type = key;
			}
		}

		auto info = ClassDB::GetInfo(std::hash<std::string>()(type));
		Ref<AssetImporter> importer = std::dynamic_pointer_cast<AssetImporter>(info.createInstance());
		importer->m_Guid = guid;
		importer->m_Type = type;
		importer->m_Path = path;
		importer->m_MetaPath = metaPath;
		importer->DeserializeMeta(node[type]);
		importer->ImportData();
		return importer;
	}
}
