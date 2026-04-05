#include "MetaSerializer.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Serialization\YamlReader.h"
#include "Blueberry\Serialization\YamlWriter.h"

#include <fstream>

namespace Blueberry
{
	void MetaSerializer::Serialize(const String& path, SerializationFlags flags)
	{
		std::ofstream stream(path.data(), std::ios::out | std::ofstream::binary);
		if (stream.is_open())
		{
			SerializationTree tree = {};
			tree.isText = true;
			SerializationNodeRef root = tree.GetRoot();
			root["Guid"] << m_Guid;
			if (m_ObjectsToSerialize.size() == 1)
			{
				Object* object = ObjectDB::GetObject(m_ObjectsToSerialize[0]);
				SerializationNodeRef objectNode = root[object->GetTypeName().c_str()];
				objectNode |= SerializationTreeFlags::MAP;
				SerializeNode(objectNode, Context::Create(object, object->GetType()), flags);
			}
			m_Trees.push_back(std::move(tree));
			YamlWriter::Write(m_Trees, stream, false);
			stream.close();
		}
	}

	void MetaSerializer::Deserialize(const String& path, SerializationFlags flags)
	{
		std::ifstream stream(path.data(), std::ios::in | std::ofstream::binary);
		if (stream.is_open())
		{
			YamlReader::Read(m_Trees, stream, false);
			SerializationTree& tree = m_Trees[0];
			SerializationNodeConstRef root = tree.GetConstRoot();
			SerializationNodeConstRef guidNode = root[0ull];
			SerializationNodeConstRef importerNode = root[1ull];

			if (guidNode.IsValid() && importerNode.IsValid())
			{
				guidNode >> m_Guid;

				String typeName = importerNode.Get().key;
				const ClassInfo* info = ClassDB::GetInfo(typeName);
				if (info == nullptr)
				{
					BB_ERROR("Class not exists.");
					return;
				}
				// Importer is only created during first deserialization
				if (m_FileIdToObjectId.size() == 0)
				{
					Object* instance = info->Create();
					m_DeserializedObjects.push_back(std::make_pair(instance->GetObjectId(), 0));
				}
				else
				{
					Object* instance = ObjectDB::GetObject(m_FileIdToObjectId.begin()->second);
					DeserializeNode(importerNode, Context::Create(instance, instance->GetType()));
				}
			}
			stream.close();
		}
	}
}
