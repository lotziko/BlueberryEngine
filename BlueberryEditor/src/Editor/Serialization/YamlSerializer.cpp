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
					const ClassInfo* info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
					if (info == nullptr)
					{
						BB_ERROR("Class not exists.");
						continue;
					}
					Object* instance = info->createInstance();
					AddDeserializedObject(instance, fileId);
					deserializedNodes.push_back({ i, instance });
				}
				else
				{
					Object* instance = it->second;
					AddDeserializedObject(instance, fileId);
					deserializedNodes.push_back({ i, instance });
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

		for (auto& field : context.info->fields)
		{
			key = ryml::to_csubstr(field.name.data());
			void* ptr = context.ptr;

			switch (field.type)
			{
			case BindingType::Bool:
				objectNode[key] << *field.Get<bool>(ptr);
				break;
			case BindingType::Int:
				objectNode[key] << *field.Get<int>(ptr);
				break;
			case BindingType::Uint:
				objectNode[key] << *field.Get<unsigned int>(ptr);
				break;
			case BindingType::Float:
				objectNode[key] << *field.Get<float>(ptr);
				break;
			case BindingType::String:
				objectNode[key] << *field.Get<String>(ptr);
				break;
			case BindingType::ByteData:
			{
				ByteData data = *field.Get<ByteData>(ptr);
				if (data.size() > 0)
				{
					DataWrapper<ByteData> wrapper = { data };
					objectNode[key] << wrapper;
				}
			}
			break;
			case BindingType::IntList:
			{
				List<int> data = *field.Get<List<int>>(ptr);
				DataWrapper<List<int>> wrapper = { data };
				objectNode[key] << wrapper;
			}
			break;
			case BindingType::FloatList:
			{
				List<float> data = *field.Get<List<float>>(ptr);
				DataWrapper<List<float>> wrapper = { data };
				objectNode[key] << wrapper;
			}
			break;
			case BindingType::StringList:
			{
				List<String> arrayValue = *field.Get<List<String>>(ptr);
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
				objectNode[key] << *field.Get<int>(ptr);
				break;
			case BindingType::Vector2:
				objectNode[key] << *field.Get<Vector2>(ptr);
				break;
			case BindingType::Vector3:
				objectNode[key] << *field.Get<Vector3>(ptr);
				break;
			case BindingType::Vector4:
				objectNode[key] << *field.Get<Vector4>(ptr);
				break;
			case BindingType::Quaternion:
				objectNode[key] << *field.Get<Quaternion>(ptr);
				break;
			case BindingType::Color:
				objectNode[key] << *field.Get<Color>(ptr);
				break;
			case BindingType::Raw:
			{
				ByteData byteData;
				byteData.resize(field.options.size);
				memcpy(byteData.data(), field.Get<uint8_t>(ptr), field.options.size);
				DataWrapper<ByteData> wrapper = { byteData };
				objectNode[key] << wrapper;
			}
			break;
			case BindingType::ObjectPtr:
			{
				ObjectPtr<Object> objectRefValue = *field.Get<ObjectPtr<Object>>(ptr);
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
				List<ObjectPtr<Object>> arrayValue = *field.Get<List<ObjectPtr<Object>>>(ptr);
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
				Data* data = field.Get<Data>(ptr);
				Context context = Context::CreateNoOffset(data, field.options.objectType);
				SerializeNode(objectNode, context);
			}
			break;
			case BindingType::DataList:
			{
				ListBase* dataArrayPointer = field.Get<ListBase>(ptr);
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
			case BindingType::Variant:
			{
				Variant variant = *field.Get<Variant>(ptr);
				ryml::NodeRef node = objectNode[key];
				node |= ryml::MAP;
				node |= ryml::_WIP_STYLE_FLOW_SL;
				node["type"] << variant.index();

				key = "data";
				switch (variant.index())
				{
				case 0:
					node[key] << std::get<bool>(variant);
					break;
				case 1:
					node[key] << std::get<int32_t>(variant);
					break;
				case 2:
					node[key] << std::get<uint32_t>(variant);
					break;
				case 3:
					node[key] << std::get<int64_t>(variant);
					break;
				case 4:
					node[key] << std::get<uint64_t>(variant);
					break;
				case 5:
					node[key] << std::get<float>(variant);
					break;
				case 6:
					node[key] << std::get<String>(variant);
					break;
				case 7:
					node[key] << std::get<Vector2>(variant);
					break;
				case 8:
					node[key] << std::get<Vector2Int>(variant);
					break;
				case 9:
					node[key] << std::get<Vector3>(variant);
					break;
				case 10:
					node[key] << std::get<Vector3Int>(variant);
					break;
				case 11:
					node[key] << std::get<Vector4>(variant);
					break;
				case 12:
					node[key] << std::get<Vector4Int>(variant);
					break;
				case 13:
					node[key] << std::get<Quaternion>(variant);
					break;
				case 14:
					node[key] << std::get<Color>(variant);
					break;
				case 15:
					node[key] << GetPtrData(std::get<Blueberry::ObjectPtr<Object>>(variant).Get());
					break;
				default:
					BB_INFO("Can't serialize field " << field.name);
					break;
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

		for (auto& field : context.info->fields)
		{
			key = ryml::to_csubstr(field.name.data());
			void* ptr = context.ptr;

			if (objectNode.has_child(key))
			{
				switch (field.type)
				{
				case BindingType::Bool:
					objectNode[key] >> *field.Get<bool>(ptr);
					break;
				case BindingType::Int:
					objectNode[key] >> *field.Get<int>(ptr);
					break;
				case BindingType::Uint:
					objectNode[key] >> *field.Get<unsigned int>(ptr);
					break;
				case BindingType::Float:
					objectNode[key] >> *field.Get<float>(ptr);
					break;
				case BindingType::String:
					objectNode[key] >> *field.Get<String>(ptr);
					break;
				case BindingType::ByteData:
				{
					ByteData data;
					DataWrapper<ByteData> wrapper = { data };
					objectNode[key] >> wrapper;
					*field.Get<ByteData>(ptr) = std::move(data);
				}
				break;
				case BindingType::IntList:
				{
					List<int> data;
					DataWrapper<List<int>> wrapper = { data };
					objectNode[key] >> wrapper;
					*field.Get<List<int>>(ptr) = std::move(data);
				}
				break;
				case BindingType::FloatList:
				{
					List<float> data;
					DataWrapper<List<float>> wrapper = { data };
					objectNode[key] >> wrapper;
					*field.Get<List<float>>(ptr) = std::move(data);
				}
				break;
				case BindingType::StringList:
				{
					List<String>* arrayPointer = field.Get<List<String>>(ptr);
					arrayPointer->clear_base();
					for (auto& child : objectNode[key].children())
					{
						String stringValue;
						child >> stringValue;
						arrayPointer->push_back(stringValue);
					}
				}
				break;
				case BindingType::Enum:
					objectNode[key] >> *field.Get<int>(ptr);
					break;
				case BindingType::Vector2:
					objectNode[key] >> *field.Get<Vector2>(ptr);
					break;
				case BindingType::Vector3:
					objectNode[key] >> *field.Get<Vector3>(ptr);
					break;
				case BindingType::Vector4:
					objectNode[key] >> *field.Get<Vector4>(ptr);
					break;
				case BindingType::Quaternion:
					objectNode[key] >> *field.Get<Quaternion>(ptr);
					break;
				case BindingType::Color:
					objectNode[key] >> *field.Get<Color>(ptr);
					break;
				case BindingType::Raw:
				{
					ByteData byteData;
					DataWrapper<ByteData> wrapper = { byteData };
					objectNode[key] >> wrapper;
					memcpy(field.Get<uint8_t>(ptr), byteData.data(), byteData.size());
				}
				break;
				case BindingType::ObjectPtr:
				{
					ObjectPtrData data = {};
					objectNode[key] >> data;
					Object* obj = GetPtrObject(data);
					*field.Get<ObjectPtr<Object>>(ptr) = obj;
				}
				break;
				case BindingType::ObjectPtrList:
				{
					List<ObjectPtr<Object>>* refArrayPointer = field.Get<List<ObjectPtr<Object>>>(ptr);
					refArrayPointer->clear_base();
					for (auto& child : objectNode[key].cchildren())
					{
						ObjectPtrData data = {};
						child >> data;
						refArrayPointer->push_back(GetPtrObject(data));
					}
				}
				break;
				case BindingType::Data:
				{
					Data* data = field.Get<Data>(ptr);
					Context context = Context::CreateNoOffset(data, field.options.objectType);
					DeserializeNode(objectNode, context);
				}
				break;
				case BindingType::DataList:
				{
					ListBase* dataArrayPointer = field.Get<ListBase>(ptr);
					dataArrayPointer->clear_base();
					for (auto& child : objectNode[key].children())
					{
						void* data = dataArrayPointer->emplace_back_base();
						Context context = Context::CreateNoOffset(data, field.options.objectType);
						DeserializeNode(child, context);
					}
				}
				break;
				case BindingType::Variant:
				{
					Variant* variant = field.Get<Variant>(ptr);
					ryml::NodeRef node = objectNode[key];
					size_t index;
					node["type"] >> index;

					key = "data";
					switch (index)
					{
					case 0:
					{
						bool value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 1:
					{
						int32_t value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 2:
					{
						uint32_t value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 3:
					{
						int64_t value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 4:
					{
						uint64_t value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 5:
					{
						float value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 6:
					{
						String value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 7:
					{
						Vector2 value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 8:
					{
						Vector2Int value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 9:
					{
						Vector3 value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 10:
					{
						Vector3Int value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 11:
					{
						Vector4 value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 12:
					{
						Vector4Int value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 13:
					{
						Quaternion value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 14:
					{
						Color value;
						node[key] >> value;
						*variant = value;
					}
					break;
					case 15:
					{
						ObjectPtrData data = {};
						node[key] >> data;
						Object* obj = GetPtrObject(data);
						*variant = ObjectPtr<Object>(obj);
					}
					break;
					default:
						BB_INFO("Can't deserialize field " << field.name);
						break;
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
