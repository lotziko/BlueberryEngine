#include "bbpch.h"
#include "Serializer.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	Serializer::Serializer(const ryml::NodeRef& root)
	{
		m_Root = root;
	}

	void Serializer::Serialize(const Ref<Object>& object)
	{
		FileId id = GetFileId(object.get());
		ryml::NodeRef objectNode = m_Root.append_child() << ryml::key(object->ToString());
		objectNode.set_key_anchor(ryml::to_csubstr(*(m_Anchors.push_back(std::to_string(id)))));
		objectNode |= ryml::MAP;
		
		ryml::csubstr key;
		Variant value;

		for (auto& field : ClassDB::GetInfo(object->GetType()).fields)
		{
			key = ryml::to_csubstr(field.name);
			field.bind->Get(object.get(), value);

			switch (field.type)
			{
			case BindingType::String:
				objectNode[key] << (std::string)value;
				break;
			case BindingType::Vector3:
				objectNode[key] << (Vector3)value;
				break;
			case BindingType::Quaternion:
				objectNode[key] << (Quaternion)value;
				break;
			case BindingType::Object:
				Object* objectValue = (Object*)value;
				if (objectValue != nullptr)
				{
					objectNode[key] << GetFileId(objectValue);
				}
				break;
			}
		}
	}

	FileId Serializer::GetFileId(Object* object)
	{
		auto idIt = m_ObjectToFileId.find(object);
		if (idIt != m_ObjectToFileId.end())
		{
			return idIt->second;
		}

		FileId id = ++m_MaxId;
		m_ObjectToFileId.insert({ object, id });
		return id;
	}
}
