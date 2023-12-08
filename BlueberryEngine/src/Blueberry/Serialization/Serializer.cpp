#include "bbpch.h"
#include "Serializer.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\YamlSerializers.h"

namespace Blueberry
{
	Serializer::Serializer(const ryml::NodeRef& root)
	{
		m_Root = root;
	}

	const ryml::NodeRef& Serializer::GetRoot()
	{
		return m_Root;
	}

	void Serializer::AddObject(Object* object)
	{
		m_ObjectToFileId.insert({ object, ++m_MaxId });
		m_ObjectsToSerialize.emplace_back(object);
	}

	void Serializer::AddObject(ryml::ConstNodeRef& node)
	{
		FileId fileId;
		if (node.has_key_tag() && ryml::from_chars(node.key_tag().trim("!"), &fileId))
		{
			ryml::csubstr key = node.key();
			std::string typeName(key.str, key.size());
			ClassDB::ClassInfo info = ClassDB::GetInfo(std::hash<std::string>()(typeName));
			Ref<Object> instance = info.createInstance();
			m_FileIdToObject.insert({ fileId, { instance, fileId - 1 } });
			m_DeserializedObjects.emplace_back(instance);
		}
	}

	void Serializer::Serialize()
	{
		for (auto& object : m_ObjectsToSerialize)
		{
			Serialize(object);
		}
	}

	void Serializer::Deserialize()
	{
		for (auto& pair : m_FileIdToObject)
		{
			Deserialize(pair.second.first, pair.second.second);
		}
	}

	std::vector<Ref<Object>>& Serializer::GetDeserializedObjects()
	{
		return m_DeserializedObjects;
	}

	void Serializer::Serialize(Object* object)
	{
		FileId id = GetFileId(object);
		ryml::NodeRef objectNode = m_Root.append_child() << ryml::key(object->GetTypeName());
		objectNode.set_key_tag(ryml::to_csubstr(*(m_Anchors.push_back(std::to_string(id)))));
		objectNode |= ryml::MAP;

		ryml::csubstr key;
		Variant value;

		for (auto& field : ClassDB::GetInfo(object->GetType()).fields)
		{
			key = ryml::to_csubstr(field.name);
			field.bind->Get(object, value);

			switch (field.type)
			{
			case BindingType::String:
				objectNode[key] << *value.Get<std::string>();
				break;
			case BindingType::Vector3:
				objectNode[key] << *value.Get<Vector3>();
				break;
			case BindingType::Vector4:
				objectNode[key] << *value.Get<Vector4>();
				break;
			case BindingType::Quaternion:
				objectNode[key] << *value.Get<Quaternion>();
				break;
			case BindingType::Color:
				objectNode[key] << *value.Get<Color>();
				break;
			case BindingType::Object:
			{
				Object* objectValue = *value.Get<Object*>();
				if (objectValue != nullptr)
				{
					objectNode[key] << GetFileId(objectValue);
				}
			}
			break;
			case BindingType::ObjectRef:
			{
				Ref<Object> objectRefValue = *value.Get<Ref<Object>>();
				if (objectRefValue != nullptr)
				{
					objectNode[key] << GetFileId(objectRefValue.get());
				}
			}
			break;
			case BindingType::ObjectPointerArray:
			{
				std::vector<Object*> arrayValue = *value.Get<std::vector<Object*>>();
				if (arrayValue.size() > 0)
				{
					ryml::NodeRef sequence = objectNode[key];
					sequence |= ryml::SEQ;
					for (Object* object : arrayValue)
					{
						sequence.append_child() << GetFileId(object);
					}
				}
			}
			break;
			case BindingType::ObjectRefArray:
			{
				std::vector<Ref<Object>> arrayValue = *value.Get<std::vector<Ref<Object>>>();
				if (arrayValue.size() > 0)
				{
					ryml::NodeRef sequence = objectNode[key];
					sequence |= ryml::SEQ;
					for (Ref<Object>& object : arrayValue)
					{
						sequence.append_child() << GetFileId(object.get());
					}
				}
			}
			break;
			default:
				continue;
			}
		}
	}

	void Serializer::Deserialize(Ref<Object>& object, int nodeIndex)
	{
		ryml::ConstNodeRef objectNode = m_Root[nodeIndex];

		ryml::csubstr key;
		Variant value;

		auto fields = ClassDB::GetInfo(object->GetType()).fields;
		for (auto& field : fields)
		{
			key = ryml::to_csubstr(field.name);
			field.bind->Get(object.get(), value);

			if (objectNode.has_child(key))
			{
				switch (field.type)
				{
				case BindingType::String:
					objectNode[key] >> *value.Get<std::string>();
					break;
				case BindingType::Vector3:
					objectNode[key] >> *value.Get<Vector3>();
					break;
				case BindingType::Vector4:
					objectNode[key] >> *value.Get<Vector4>();
					break;
				case BindingType::Quaternion:
					objectNode[key] >> *value.Get<Quaternion>();
					break;
				case BindingType::Color:
					objectNode[key] >> *value.Get<Color>();
					break;
				case BindingType::Object:
				{
					FileId fileId;
					objectNode[key] >> fileId;
					*value.Get<Object*>() = GetObjectRef(fileId).get();
				}
				break;
				case BindingType::ObjectRef:
				{
					FileId fileId;
					objectNode[key] >> fileId;
					*value.Get<Ref<Object>>() = GetObjectRef(fileId);
				}
				break;
				case BindingType::ObjectRefArray:
				{
					FileId fileId;
					std::vector<Ref<Object>>* refArrayPointer = value.Get<std::vector<Ref<Object>>>();
					for (auto& child : objectNode[key].cchildren())
					{
						child >> fileId;
						refArrayPointer->emplace_back(GetObjectRef(fileId));
					}
				}
				break;
				default:
					continue;
				}
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

	Ref<Object> Serializer::GetObjectRef(const FileId& fileId)
	{
		auto idIt = m_FileIdToObject.find(fileId);
		if (idIt != m_FileIdToObject.end())
		{
			return idIt->second.first;
		}

		return nullptr;
	}
}
