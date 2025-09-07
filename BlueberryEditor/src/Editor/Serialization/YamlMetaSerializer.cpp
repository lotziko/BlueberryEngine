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

	void YamlMetaSerializer::Serialize(const String& path)
	{
		m_AssetGuid = ObjectDB::GetGuidFromObject(m_ObjectsToSerialize[0]);
		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;
		root["Guid"] << m_Guid;
		if (m_ObjectsToSerialize.size() == 1)
		{
			Object* object = m_ObjectsToSerialize[0];
			ryml::NodeRef objectNode = root.append_child() << ryml::key(object->GetTypeName());
			objectNode |= ryml::MAP;
			SerializeNode(objectNode, Context::Create(object, object->GetType()));
		}
		YamlHelper::Save(tree, path);
	}

	void YamlMetaSerializer::Deserialize(const String& path)
	{
		ryml::Tree tree;
		YamlHelper::Load(tree, path);
		ryml::NodeRef root = tree.rootref();
		root[0] >> m_Guid;
		ryml::ConstNodeRef node = root[1];
		ryml::csubstr key = node.key();
		String typeName(key.str, key.size());
		ClassInfo info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
		// Importer is only created during first deserialization
		if (m_FileIdToObject.size() == 0)
		{
			Object* instance = info.createInstance();
			m_DeserializedObjects.emplace_back(std::make_pair(instance, 0));
		}
		else
		{
			Object* instance = m_FileIdToObject.begin()->second;
			DeserializeNode(root[1], Context::Create(instance, instance->GetType()));
		}
	}
}
