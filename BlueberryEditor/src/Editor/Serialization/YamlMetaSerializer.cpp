#include "bbpch.h"
#include "YamlMetaSerializer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	const Guid& YamlMetaSerializer::GetGuid()
	{
		return m_Guid;
	}

	void YamlMetaSerializer::SetGuid(const Guid& guid)
	{
		m_Guid = guid;
	}

	void YamlMetaSerializer::Serialize(const std::string& path)
	{
		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;
		root["Guid"] << m_Guid;
		if (m_ObjectsToSerialize.size() == 1)
		{
			Object* object = m_ObjectsToSerialize[0];
			ryml::NodeRef objectNode = root.append_child() << ryml::key(object->GetTypeName());
			objectNode |= ryml::MAP;
			SerializeNode(objectNode, object);
		}
		YamlHelper::Save(tree, path);
	}

	void YamlMetaSerializer::Deserialize(const std::string& path)
	{
		ryml::Tree tree;
		YamlHelper::Load(tree, path);
		ryml::NodeRef root = tree.rootref();
		root[0] >> m_Guid;
		ryml::ConstNodeRef node = root[1];
		ryml::csubstr key = node.key();
		std::string typeName(key.str, key.size());
		ClassDB::ClassInfo info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
		Object* instance = info.createInstance();
		m_DeserializedObjects.emplace_back(std::pair { instance, 0 });
		DeserializeNode(root[1], instance);
	}
}
