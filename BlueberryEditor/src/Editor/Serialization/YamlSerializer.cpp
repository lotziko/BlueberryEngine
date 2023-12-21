#include "bbpch.h"
#include "YamlSerializer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\WeakObjectPtr.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	void YamlSerializer::Serialize(const std::string& path)
	{
		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;
		for (auto& object : m_ObjectsToSerialize)
		{
			ryml::NodeRef objectNode = root.append_child() << ryml::key(object->GetTypeName());
			objectNode |= ryml::MAP;
			FileId id = GetFileId(object);
			objectNode.set_key_tag(ryml::to_csubstr(*(m_Tags.push_back(std::to_string(id)))));
			SerializeNode(objectNode, object);
		}
		YamlHelper::Save(tree, path);
	}

	void YamlSerializer::Deserialize(const std::string& path)
	{
		ryml::Tree tree;
		YamlHelper::Load(tree, path);
		ryml::NodeRef root = tree.rootref();
		std::vector<std::pair<int, Object*>> deserializedNodes;
		for (size_t i = 0; i < root.num_children(); i++)
		{
			ryml::ConstNodeRef node = root[i];
			FileId fileId;
			if (node.has_key_tag() && ryml::from_chars(node.key_tag().trim("!"), &fileId))
			{
				ryml::csubstr key = node.key();
				std::string typeName(key.str, key.size());
				ClassDB::ClassInfo info = ClassDB::GetInfo(std::hash<std::string>()(typeName));
				Object* instance = info.createInstance();
				m_FileIdToObject.insert({ fileId, instance });
				m_DeserializedObjects.emplace_back(instance);
				deserializedNodes.emplace_back(i, instance);
			}
		}
		for (auto& pair : deserializedNodes)
		{
			DeserializeNode(root[pair.first], pair.second);
		}
	}

	void YamlSerializer::SerializeNode(ryml::NodeRef& objectNode, Object* object)
	{
		ryml::csubstr key;
		Variant value;

		for (auto& field : ClassDB::GetInfo(object->GetType()).fields)
		{
			key = ryml::to_csubstr(field.name);
			field.bind->Get(object, value);

			switch (field.type)
			{
			case BindingType::Int:
				objectNode[key] << *value.Get<int>();
				break;
			case BindingType::Float:
				objectNode[key] << *value.Get<float>();
				break;
			case BindingType::String:
				objectNode[key] << *value.Get<std::string>();
				break;
			case BindingType::ByteData:
				objectNode[key] << *value.Get<ByteData>();
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
			case BindingType::ObjectPtr:
			{
				Object* objectValue = *value.Get<Object*>();
				if (objectValue != nullptr)
				{
					objectNode[key] << GetFileId(objectValue);
				}
			}
			break;
			case BindingType::ObjectWeakPtr:
			{
				WeakObjectPtr<Object> objectRefValue = *value.Get<WeakObjectPtr<Object>>();
				if (objectRefValue.IsValid())
				{
					objectNode[key] << GetFileId(objectRefValue.Get());
				}
			}
			break;
			case BindingType::ObjectPtrArray:
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
			case BindingType::ObjectWeakPtrArray:
			{
				std::vector<WeakObjectPtr<Object>> arrayValue = *value.Get<std::vector<WeakObjectPtr<Object>>>();
				if (arrayValue.size() > 0)
				{
					ryml::NodeRef sequence = objectNode[key];
					sequence |= ryml::SEQ;
					for (WeakObjectPtr<Object>& object : arrayValue)
					{
						if (object.IsValid())
						{
							sequence.append_child() << GetFileId(object.Get());
						}
					}
				}
			}
			break;
			default:
				continue;
			}
		}
	}

	void YamlSerializer::DeserializeNode(ryml::NodeRef& objectNode, Object* object)
	{
		ryml::csubstr key;
		Variant value;

		auto fields = ClassDB::GetInfo(object->GetType()).fields;
		for (auto& field : fields)
		{
			key = ryml::to_csubstr(field.name);
			field.bind->Get(object, value);

			if (objectNode.has_child(key))
			{
				switch (field.type)
				{
				case BindingType::Int:
					objectNode[key] >> *value.Get<int>();
					break;
				case BindingType::Float:
					objectNode[key] >> *value.Get<float>();
					break;
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
				case BindingType::ObjectPtr:
				{
					FileId fileId;
					objectNode[key] >> fileId;
					*value.Get<Object*>() = GetObjectRef(fileId);
				}
				break;
				case BindingType::ObjectWeakPtr:
				{
					FileId fileId;
					objectNode[key] >> fileId;
					*value.Get<WeakObjectPtr<Object>>() = GetObjectRef(fileId);
				}
				break;
				case BindingType::ObjectWeakPtrArray:
				{
					FileId fileId;
					std::vector<WeakObjectPtr<Object>>* refArrayPointer = value.Get<std::vector<WeakObjectPtr<Object>>>();
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
}
