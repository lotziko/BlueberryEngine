#include "bbpch.h"

#include "AssetImporter.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\YamlHelper.h"
#include "Blueberry\Serialization\Serializer.h"

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
		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;
		Serializer serializer(root);

		/*auto importerType = m_Type.c_str();
		root["Guid"] << m_Guid;
		ryml::NodeRef dataNode = root.append_child() << ryml::key(importerType);
		dataNode |= ryml::MAP;
		Serialize(context, dataNode);

		YamlHelper::Save(tree, m_MetaPath);*/
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
		ryml::Tree tree;
		YamlHelper::Load(tree, metaPath);

		ryml::NodeRef root = tree.rootref();

		Guid guid;
		root[0] >> guid;
		auto importerKey = root[1].key();
		std::string type(importerKey.str, importerKey.len);

		auto info = ClassDB::GetInfo(std::hash<std::string>()(type));
		Ref<AssetImporter> importer = std::dynamic_pointer_cast<AssetImporter>(info.createInstance());
		importer->m_Guid = guid;
		importer->m_Type = type;
		importer->m_Path = path;
		importer->m_MetaPath = metaPath;
		//importer->Deserialize(root[1]);
		importer->ImportData();
		return importer;
	}

	void AssetImporter::BindProperties()
	{
	}
}
