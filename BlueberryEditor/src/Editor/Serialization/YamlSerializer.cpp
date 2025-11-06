#include "YamlSerializer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Variant.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Editor\Serialization\YamlHelper.h"
#include "Editor\Serialization\YamlSerializers.h"

namespace Blueberry
{
	void YamlSerializer::Serialize(const String& path)
	{
		m_AssetGuid = ObjectDB::GetGuidFromObject(m_ObjectsToSerialize[0]);
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
			SerializeNode(objectNode, Context::Create(object, object->GetType()));
		}
		YamlHelper::Save(tree, path);
	}

	void YamlSerializer::Deserialize(const String& path)
	{
		ryml::Tree tree;
		YamlHelper::Load(tree, path);
		ryml::NodeRef root = tree.rootref();
		List<std::pair<int, Object*>> deserializedNodes;
		for (size_t i = 0; i < root.num_children(); i++)
		{
			ryml::ConstNodeRef node = root[i];
			FileId fileId;
			if (node.has_key_tag() && ryml::from_chars(node.key_tag().trim("!"), &fileId))
			{
				auto it = m_FileIdToObject.find(fileId);
				if (it == m_FileIdToObject.end())
				{
					ryml::csubstr key = node.key();
					String typeName(key.str, key.size());
					ClassInfo info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
					Object* instance = info.createInstance();
					AddDeserializedObject(instance, fileId);
					deserializedNodes.emplace_back(i, instance);
				}
				else
				{
					Object* instance = it->second;
					AddDeserializedObject(instance, fileId);
					deserializedNodes.emplace_back(i, instance);
				}
			}
		}
		for (auto& pair : deserializedNodes)
		{
			Object* object = pair.second;
			DeserializeNode(root[pair.first], Context::Create(object, object->GetType()));
		}
	}

	void YamlSerializer::SerializeNode(ryml::NodeRef& objectNode, Context context)
	{
		ryml::csubstr key;
		Variant value;

		for (auto& field : context.info.fields)
		{
			key = ryml::to_csubstr(field.name.data());
			value = Variant(context.ptr, field.offset);

			switch (field.type)
			{
			case BindingType::Bool:
				objectNode[key] << *value.Get<bool>();
				break;
			case BindingType::Int:
				objectNode[key] << *value.Get<int>();
				break;
			case BindingType::Float:
				objectNode[key] << *value.Get<float>();
				break;
			case BindingType::String:
				objectNode[key] << *value.Get<String>();
				break;
			case BindingType::ByteData:
			{
				ByteData data = *value.Get<ByteData>();
				if (data.size() > 0)
				{
					DataWrapper<ByteData> wrapper = { data };
					objectNode[key] << wrapper;
				}
			}
			break;
			case BindingType::IntList:
			{
				List<int> data = *value.Get<List<int>>();
				DataWrapper<List<int>> wrapper = { data };
				objectNode[key] << wrapper;
			}
			break;
			case BindingType::FloatList:
			{
				List<float> data = *value.Get<List<float>>();
				DataWrapper<List<float>> wrapper = { data };
				objectNode[key] << wrapper;
			}
			break;
			case BindingType::StringList:
			{
				List<String> arrayValue = *value.Get<List<String>>();
				if (arrayValue.size() > 0)
				{
					ryml::NodeRef sequence = objectNode[key];
					sequence |= ryml::SEQ;
					for (auto stringValue : arrayValue)
					{
						sequence.append_child() << stringValue;
					}
				}
			}
			break;
			case BindingType::Enum:
				objectNode[key] << *value.Get<int>();
				break;
			case BindingType::Vector2:
				objectNode[key] << *value.Get<Vector2>();
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
			case BindingType::Raw:
			{
				ByteData byteData;
				byteData.resize(field.options.size);
				memcpy(byteData.data(), value.Get<uint8_t>(), field.options.size);
				DataWrapper<ByteData> wrapper = { byteData };
				objectNode[key] << wrapper;
			}
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
			case BindingType::ObjectPtrList:
			{
				List<ObjectPtr<Object>> arrayValue = *value.Get<List<ObjectPtr<Object>>>();
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
			case BindingType::Data:
			{
				Data* data = value.Get<Data>();
				Context context = Context::CreateNoOffset(data, field.options.objectType);
				SerializeNode(objectNode, context);
			}
			break;
			case BindingType::DataList:
			{
				ListBase* dataArrayPointer = value.Get<ListBase>();
				uint32_t dataSize = static_cast<uint32_t>(dataArrayPointer->size_base());
				if (dataSize > 0)
				{
					ryml::NodeRef sequence = objectNode[key];
					sequence |= ryml::SEQ;
					for (uint32_t i = 0; i < dataSize; ++i)
					{
						void* data = dataArrayPointer->get_base(i);
						Context context = Context::CreateNoOffset(data, field.options.objectType);
						ryml::NodeRef node = sequence.append_child();
						node |= ryml::MAP;
						SerializeNode(node, context);
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

	void YamlSerializer::DeserializeNode(ryml::NodeRef& objectNode, Context context)
	{
		ryml::csubstr key;
		Variant value;

		auto fields = context.info.fields;
		for (auto& field : fields)
		{
			key = ryml::to_csubstr(field.name.data());
			value = Variant(context.ptr, field.offset);

			if (objectNode.has_child(key))
			{
				switch (field.type)
				{
				case BindingType::Bool:
					objectNode[key] >> *value.Get<bool>();
					break;
				case BindingType::Int:
					objectNode[key] >> *value.Get<int>();
					break;
				case BindingType::Float:
					objectNode[key] >> *value.Get<float>();
					break;
				case BindingType::String:
					objectNode[key] >> *value.Get<String>();
					break;
				case BindingType::ByteData:
				{
					ByteData data;
					DataWrapper<ByteData> wrapper = { data };
					objectNode[key] >> wrapper;
					*value.Get<ByteData>() = std::move(data);
				}
				break;
				case BindingType::IntList:
				{
					List<int> data;
					DataWrapper<List<int>> wrapper = { data };
					objectNode[key] >> wrapper;
					*value.Get<List<int>>() = std::move(data);
				}
				break;
				case BindingType::FloatList:
				{
					List<float> data;
					DataWrapper<List<float>> wrapper = { data };
					objectNode[key] >> wrapper;
					*value.Get<List<float>>() = std::move(data);
				}
				break;
				case BindingType::StringList:
				{
					List<String>* arrayPointer = value.Get<List<String>>();
					for (auto& child : objectNode[key].children())
					{
						String stringValue;
						child >> stringValue;
						arrayPointer->emplace_back(stringValue);
					}
				}
				break;
				case BindingType::Enum:
					objectNode[key] >> *value.Get<int>();
					break;
				case BindingType::Vector2:
					objectNode[key] >> *value.Get<Vector2>();
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
				case BindingType::Raw:
				{
					ByteData byteData;
					DataWrapper<ByteData> wrapper = { byteData };
					objectNode[key] >> wrapper;
					memcpy(value.Get<uint8_t>(), byteData.data(), byteData.size());
				}
				break;
				case BindingType::ObjectPtr:
				{
					ObjectPtrData data = {};
					objectNode[key] >> data;
					Object* obj = GetPtrObject(data);
					*value.Get<ObjectPtr<Object>>() = obj;
				}
				break;
				case BindingType::ObjectPtrList:
				{
					List<ObjectPtr<Object>>* refArrayPointer = value.Get<List<ObjectPtr<Object>>>();
					for (auto& child : objectNode[key].cchildren())
					{
						ObjectPtrData data = {};
						child >> data;
						refArrayPointer->emplace_back(GetPtrObject(data));
					}
				}
				break;
				case BindingType::Data:
				{
					Data* data = value.Get<Data>();
					Context context = Context::CreateNoOffset(data, field.options.objectType);
					DeserializeNode(objectNode, context);
				}
				break;
				case BindingType::DataList:
				{
					ListBase* dataArrayPointer = value.Get<ListBase>();
					for (auto& child : objectNode[key].children())
					{
						void* data = dataArrayPointer->emplace_back_base();
						Context context = Context::CreateNoOffset(data, field.options.objectType);
						DeserializeNode(child, context);
					}
				}
				break;
				default:
					BB_INFO("Can't deserialize field " << field.name);
					continue;
				}
			}
		}
	}
}
