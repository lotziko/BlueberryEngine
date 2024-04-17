#include "bbpch.h"
#include "YamlSerializer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	void YamlSerializer::Serialize(const std::string& path)
	{
		ryml::Tree tree;
		ryml::NodeRef root = tree.rootref();
		root |= ryml::MAP;

		Object* object;
		while ((object = GetNextObjectToSerialize()) != nullptr)
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
				ClassDB::ClassInfo info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
				Object* instance = info.createInstance();
				AddDeserializedObject(instance, fileId);
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
			{
				ByteData data = *value.Get<ByteData>();
				if (data.size > 0)
				{
					objectNode[key] << data;
				}
			}
			break;
			case BindingType::IntByteArray:
			{
				std::vector<int> data = *value.Get<std::vector<int>>();
				ByteData byteData;
				byteData.data = (byte*)data.data();
				byteData.size = data.size() * sizeof(int);
				objectNode[key] << byteData;
			}
			break;
			case BindingType::FloatByteArray:
			{
				std::vector<float> data = *value.Get<std::vector<float>>();
				ByteData byteData;
				byteData.data = (byte*)data.data();
				byteData.size = data.size() * sizeof(float);
				objectNode[key] << byteData;
			}
			break;
			case BindingType::Enum:
				objectNode[key] << *value.Get<int>();
				break;
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
				ObjectPtr<Object> objectRefValue = *value.Get<ObjectPtr<Object>>();
				ObjectPtrData data = {};
				if (objectRefValue.IsValid())
				{
					data = GetPtrData(objectRefValue.Get());
				}
				else
				{
					data.fileId = 0;
				}
				objectNode[key] << data;
			}
			break;
			case BindingType::ObjectPtrArray:
			{
				std::vector<ObjectPtr<Object>> arrayValue = *value.Get<std::vector<ObjectPtr<Object>>>();
				if (arrayValue.size() > 0)
				{
					ryml::NodeRef sequence = objectNode[key];
					sequence |= ryml::SEQ;
					for (ObjectPtr<Object>& objectRefValue : arrayValue)
					{
						ObjectPtrData data = {};
						if (objectRefValue.IsValid())
						{
							data = GetPtrData(objectRefValue.Get());
						}
						else
						{
							data.fileId = 0;
						}
						sequence.append_child() << data;
					}
				}
			}
			break;
			default:
				BB_INFO("Can't serialize field " << field.name);
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
				case BindingType::ByteData:
					objectNode[key] >> *value.Get<ByteData>();
					break;
				case BindingType::IntByteArray:
				{
					ByteData byteData;
					objectNode[key] >> byteData;
					int* ptr = reinterpret_cast<int*>(byteData.data);
					std::vector<int> data(ptr, ptr + byteData.size / sizeof(int));
					*value.Get<std::vector<int>>() = data;
				}
				break;
				case BindingType::FloatByteArray:
				{
					ByteData byteData;
					objectNode[key] >> byteData;
					float* ptr = reinterpret_cast<float*>(byteData.data);
					std::vector<float> data(ptr, ptr + byteData.size / sizeof(float));
					*value.Get<std::vector<float>>() = data;
				}
				break;
				case BindingType::Enum:
					objectNode[key] >> *value.Get<int>();
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
					ObjectPtrData data = {};
					objectNode[key] >> data;
					*value.Get<ObjectPtr<Object>>() = GetPtrObject(data);
				}
				break;
				case BindingType::ObjectPtrArray:
				{
					std::vector<ObjectPtr<Object>>* refArrayPointer = value.Get<std::vector<ObjectPtr<Object>>>();
					for (auto& child : objectNode[key].cchildren())
					{
						ObjectPtrData data = {};
						child >> data;
						refArrayPointer->emplace_back(GetPtrObject(data));
					}
				}
				break;
				default:
					BB_INFO("Can't deserialize field " << field.name);
					continue;
				}
			}
		}
		object->OnCreate();
	}
}
