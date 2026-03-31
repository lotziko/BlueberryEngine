#include "Blueberry\Serialization\Serializer.h"

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\Variant.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Serialization\YamlWriter.h"
#include "Blueberry\Serialization\YamlReader.h"
#include "Blueberry\Serialization\BinaryWriter.h"
#include "Blueberry\Serialization\BinaryReader.h"

#include <fstream>
#include <iomanip>
#include <random>

namespace Blueberry
{
	void Serializer::Serialize(const String& path, bool isText)
	{
		std::ofstream stream(path.data(), std::ios::out | std::ofstream::binary);
		if (stream.is_open())
		{
			while ((m_CurrentObject = GetNextObjectToSerialize()) != nullptr)
			{
				SerializationTree tree = {};
				tree.typeId = m_CurrentObject->GetType();
				tree.fileId = GetFileId(m_CurrentObject->GetObjectId());
				tree.objectId = m_CurrentObject->GetObjectId();
				tree.isText = isText;
				SerializeNode(tree.GetRoot(), Context::Create(m_CurrentObject, m_CurrentObject->GetType()));
				m_Trees.push_back(std::move(tree));
			}
			if (isText)
			{
				YamlWriter::Write(m_Trees, stream, true);
			}
			else
			{
				BinaryWriter::Write(m_Trees, stream);
			}
			stream.close();
		}
	}

	void Serializer::Deserialize(const String& path)
	{
		std::ifstream stream(path.data(), std::ios::in | std::ofstream::binary);
		if (stream.is_open())
		{
			m_Trees.clear();
			switch (stream.peek())
			{
			case '%':
				YamlReader::Read(m_Trees, stream, true);
				break;
			case 'B':
				BinaryReader::Read(m_Trees, stream);
				break;
			}
			for (auto& tree : m_Trees)
			{
				Object* instance = nullptr;
				auto it = m_FileIdToObjectId.find(tree.fileId);
				if (it == m_FileIdToObjectId.end())
				{
					const ClassInfo* info = ClassDB::GetInfo(tree.typeId);
					if (info == nullptr)
					{
						BB_ERROR("Class not exists.");
						continue;
					}
					instance = info->Create();
				}
				else
				{
					instance = ObjectDB::GetObject(it->second);
				}
				tree.objectId = instance->GetObjectId();
				AddDeserializedObject(instance->GetObjectId(), tree.fileId);
			}
			for (auto& tree : m_Trees)
			{
				Object* object = ObjectDB::GetObject(tree.objectId);
				DeserializeNode(tree.GetConstRoot(), Context::Create(object, object->GetType()));
			}
			stream.close();
		}
	}

	const Guid& Serializer::GetGuid()
	{
		return m_Guid;
	}

	void Serializer::SetGuid(const Guid& guid)
	{
		m_Guid = guid;
	}

	List<std::pair<ObjectId, FileId>>& Serializer::GetDeserializedObjects()
	{
		return m_DeserializedObjects;
	}

	void Serializer::AddObject(Object* object)
	{
		ObjectId objectId = object->GetObjectId();
		FileId fileId = GetFileId(objectId);
		m_FileIdToObjectId.insert({ fileId, objectId });
		m_ObjectsToSerialize.push_back(objectId);
	}

	void Serializer::AddObject(Object* object, FileId fileId)
	{
		m_FileIdToObjectId.insert({ fileId, object->GetObjectId() });
	}

	FileId Serializer::GetFileId(ObjectId objectId)
	{
		if (ObjectDB::HasFileId(objectId))
		{
			return ObjectDB::GetFileIdFromObjectId(objectId);
		}

		FileId fileId = GenerateFileId();
		m_FileIdToObjectId.insert_or_assign(fileId, objectId);
		if (m_Guid.IsValid())
		{
			ObjectDB::AllocateIdToGuid(objectId, m_Guid, fileId);
		}
		else
		{
			ObjectDB::AllocateIdToFileId(objectId, fileId);
		}
		return fileId;
	}

	Object* Serializer::GetObjectRef(FileId fileId)
	{
		auto idIt = m_FileIdToObjectId.find(fileId);
		if (idIt != m_FileIdToObjectId.end())
		{
			return ObjectDB::GetObject(idIt->second);
		}
		return nullptr;
	}

	Object* Serializer::GetNextObjectToSerialize()
	{
		if (m_ObjectsToSerialize.size() == 0 && m_AdditionalObjectsToSerialize.size() == 0)
		{
			return nullptr;
		}

		Object* object = nullptr;
		if (m_AdditionalObjectsToSerialize.size() > 0)
		{
			object = ObjectDB::GetObject(m_AdditionalObjectsToSerialize.front());
			m_AdditionalObjectsToSerialize.erase(m_AdditionalObjectsToSerialize.begin());
		}
		else
		{
			object = ObjectDB::GetObject(m_ObjectsToSerialize[0]);
			m_ObjectsToSerialize.erase(m_ObjectsToSerialize.begin());
		}
		return object;
	}

	Object* Serializer::GetPtrObject(const ObjectPtrData& data)
	{
		Object* result = nullptr;
		if (data.guid.data[0] > 0)
		{
			result = ObjectDB::GetObjectFromGuid(data.guid, data.fileId);
			if (result == nullptr || result->GetState() == ObjectState::AwaitingLoading)
			{
				result = AssetLoader::Load(data.guid, data.fileId);
				if (result == nullptr)
				{
					ObjectDB::AllocateEmptyObjectWithGuid(data.guid, data.fileId);
					result = ObjectDB::GetObjectFromGuid(data.guid, data.fileId);
				}
			}
		}
		else
		{
			if (data.fileId == 0)
			{
				result = nullptr;
			}
			else
			{
				result = GetObjectRef(data.fileId);
			}
		}
		return result;
	}

	ObjectPtrData Serializer::GetPtrData(Object* object)
	{
		ObjectId objectId = object->GetObjectId();
		ObjectPtrData result = {};
		if (ObjectDB::HasGuid(object))
		{
			auto pair = ObjectDB::GetGuidAndFileIdFromObject(object);
			auto fileIdIt = m_FileIdToObjectId.find(pair.second);
			if (fileIdIt != m_FileIdToObjectId.end() && fileIdIt->second == objectId)
			{
				result.fileId = pair.second;
			}
			else if (m_Guid == pair.first)
			{
				result.fileId = pair.second;
				AddAdditionalObject(objectId);
			}
			else
			{
				result.guid = pair.first;
				result.fileId = pair.second;
			}
		}
		else
		{
			FileId fileId;
			if (!ObjectDB::HasFileId(object))
			{
				fileId = GetFileId(objectId);
				AddAdditionalObject(objectId);
			}
			else
			{
				fileId = ObjectDB::GetFileIdFromObject(object);
				if (m_FileIdToObjectId.count(fileId) == 0)
				{
					AddAdditionalObject(objectId);
				}
			}
			result.fileId = fileId;
		}
		return result;
	}

	FileId Serializer::GenerateFileId()
	{
		static std::random_device rd;
		static std::mt19937 gen(rd());
		static std::uniform_int_distribution<unsigned long long> dis((std::numeric_limits<std::uint64_t>::min)(), (std::numeric_limits<std::uint64_t>::max)());

		return dis(gen);
	}

	void Serializer::SerializeNode(SerializationNodeRef node, Context context)
	{
		for (auto& field : context.info->fields)
		{
			void* ptr = context.ptr;
			SerializationNodeRef fieldNode = node[field.name.c_str()];

			switch (field.type)
			{
			case BindingType::Bool:
				fieldNode << *field.Get<bool>(ptr);
				break;
			case BindingType::Int:
				fieldNode << *field.Get<int>(ptr);
				break;
			case BindingType::Uint:
				fieldNode << *field.Get<unsigned int>(ptr);
				break;
			case BindingType::Float:
				fieldNode << *field.Get<float>(ptr);
				break;
			case BindingType::String:
				fieldNode << *field.Get<String>(ptr);
				break;
			case BindingType::ByteData:
			{
				ByteData* data = field.Get<ByteData>(ptr);
				if (data->size() > 0)
				{
					DataWrapper<ByteData> wrapper = { *data };
					fieldNode << wrapper;
				}
			}
			break;
			case BindingType::IntList:
			{
				List<int>* data = field.Get<List<int>>(ptr);
				DataWrapper<List<int>> wrapper = { *data };
				fieldNode << wrapper;
			}
			break;
			case BindingType::FloatList:
			{
				List<float>* data = field.Get<List<float>>(ptr);
				DataWrapper<List<float>> wrapper = { *data };
				fieldNode << wrapper;
			}
			break;
			case BindingType::StringList:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				List<String>* arrayValue = field.Get<List<String>>(ptr);
				if (arrayValue->size() > 0)
				{
					for (size_t i = 0; i < arrayValue->size(); ++i)
					{
						fieldNode.AppendChild() << arrayValue->at(i);
					}
				}
			}
			break;
			case BindingType::Enum:
				fieldNode << *field.Get<int>(ptr);
				break;
			case BindingType::Vector2:
				fieldNode << *field.Get<Vector2>(ptr);
				break;
			case BindingType::Vector2Int:
				fieldNode << *field.Get<Vector2Int>(ptr);
				break;
			case BindingType::Vector3:
				fieldNode << *field.Get<Vector3>(ptr);
				break;
			case BindingType::Vector3Int:
				fieldNode << *field.Get<Vector3Int>(ptr);
				break;
			case BindingType::Vector4:
				fieldNode << *field.Get<Vector4>(ptr);
				break;
			case BindingType::Vector4Int:
				fieldNode << *field.Get<Vector4Int>(ptr);
				break;
			case BindingType::Quaternion:
				fieldNode << *field.Get<Quaternion>(ptr);
				break;
			case BindingType::Color:
				fieldNode << *field.Get<Color>(ptr);
				break;
			case BindingType::AABB:
				fieldNode << *field.Get<AABB>(ptr);
				break;
			case BindingType::Matrix:
				fieldNode << *field.Get<Matrix>(ptr);
				break;
			case BindingType::Vector2List:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				List<Vector2>* arrayValue = field.Get<List<Vector2>>(ptr);
				for (size_t i = 0; i < arrayValue->size(); ++i)
				{
					SerializationNodeRef childNode = fieldNode.AppendChild();
					childNode |= SerializationFlags::FLOWMAP;
					childNode << arrayValue->at(i);
				}
			}
			break;
			case BindingType::Vector3List:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				List<Vector3>* arrayValue = field.Get<List<Vector3>>(ptr);
				for (size_t i = 0; i < arrayValue->size(); ++i)
				{
					SerializationNodeRef childNode = fieldNode.AppendChild();
					childNode |= SerializationFlags::FLOWMAP;
					childNode << arrayValue->at(i);
				}
			}
			break;
			case BindingType::Vector4List:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				List<Vector4>* arrayValue = field.Get<List<Vector4>>(ptr);
				for (size_t i = 0; i < arrayValue->size(); ++i)
				{
					SerializationNodeRef childNode = fieldNode.AppendChild();
					childNode |= SerializationFlags::FLOWMAP;
					childNode << arrayValue->at(i);
				}
			}
			break;
			case BindingType::QuaternionList:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				List<Quaternion>* arrayValue = field.Get<List<Quaternion>>(ptr);
				for (size_t i = 0; i < arrayValue->size(); ++i)
				{
					SerializationNodeRef childNode = fieldNode.AppendChild();
					childNode |= SerializationFlags::FLOWMAP;
					childNode << arrayValue->at(i);
				}
			}
			break;
			case BindingType::MatrixList:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				List<Matrix>* arrayValue = field.Get<List<Matrix>>(ptr);
				for (size_t i = 0; i < arrayValue->size(); ++i)
				{
					SerializationNodeRef childNode = fieldNode.AppendChild();
					childNode << arrayValue->at(i);
				}
			}
			break;
			case BindingType::Raw:
			{
				ByteData byteData(field.options.size);
				memcpy(byteData.data(), field.Get<uint8_t>(ptr), field.options.size);
				DataWrapper<ByteData> wrapper = { byteData };
				fieldNode << wrapper;
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
				fieldNode << data;
			}
			break;
			case BindingType::ObjectPtrList:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				List<ObjectPtr<Object>>* arrayValue = field.Get<List<ObjectPtr<Object>>>(ptr);
				for (size_t i = 0; i < arrayValue->size(); ++i)
				{
					ObjectPtr<Object> objectRefValue = arrayValue->at(i);
					ObjectPtrData data = {};
					if (objectRefValue.IsValid())
					{
						data = GetPtrData(objectRefValue.Get());
					}
					else
					{
						data.fileId = 0;
					}
					fieldNode.AppendChild() << data;
				}
			}
			break;
			case BindingType::Data:
			{
				fieldNode |= SerializationFlags::MAP;
				Data* data = field.Get<Data>(ptr);
				Context context = Context::CreateNoOffset(data, *field.options.objectType);
				if (context.info != nullptr)
				{
					SerializeNode(fieldNode, context);
				}
				else
				{
					BB_ERROR("Data class not exists.");
				}
			}
			break;
			case BindingType::DataList:
			{
				fieldNode |= SerializationFlags::SEQUENCE;
				ListBase* dataArrayPointer = field.Get<ListBase>(ptr);
				size_t dataSize = dataArrayPointer->size_base();
				for (size_t i = 0; i < dataSize; ++i)
				{
					void* data = dataArrayPointer->get_base(i);
					Context context = Context::CreateNoOffset(data, *field.options.objectType);
					if (context.info != nullptr)
					{
						SerializationNodeRef dataNode = fieldNode.AppendChild();
						dataNode |= SerializationFlags::MAP;
						SerializeNode(dataNode, context);
					}
					else
					{
						BB_ERROR("Data class not exists.");
					}
				}
			}
			break;
			case BindingType::Variant:
			{
				Variant variant = *field.Get<Variant>(ptr);
				size_t type = variant.index();
				fieldNode |= SerializationFlags::FLOWMAP;

				SerializationNodeRef typeNode = fieldNode["type"];
				SerializationNodeRef dataNode = fieldNode["data"];
				typeNode << type;

				switch (type)
				{
				case 0:
					dataNode << std::get<bool>(variant);
					break;
				case 1:
					dataNode << std::get<int32_t>(variant);
					break;
				case 2:
					dataNode << std::get<uint32_t>(variant);
					break;
				case 3:
					dataNode << std::get<int64_t>(variant);
					break;
				case 4:
					dataNode << std::get<uint64_t>(variant);
					break;
				case 5:
					dataNode << std::get<float>(variant);
					break;
				case 6:
					dataNode << std::get<String>(variant);
					break;
				case 7:
					dataNode << std::get<Vector2>(variant);
					break;
				case 8:
					dataNode << std::get<Vector2Int>(variant);
					break;
				case 9:
					dataNode << std::get<Vector3>(variant);
					break;
				case 10:
					dataNode << std::get<Vector3Int>(variant);
					break;
				case 11:
					dataNode << std::get<Vector4>(variant);
					break;
				case 12:
					dataNode << std::get<Vector4Int>(variant);
					break;
				case 13:
					dataNode << std::get<Quaternion>(variant);
					break;
				case 14:
					dataNode << std::get<Color>(variant);
					break;
				case 15:
					dataNode << GetPtrData(std::get<Blueberry::ObjectPtr<Object>>(variant).Get());
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

	void Serializer::DeserializeNode(SerializationNodeConstRef node, Context context)
	{
		for (auto& fieldNode : node.GetChildren())
		{
			void* ptr = context.ptr;
			const FieldInfo* field = context.info->GetField(fieldNode.Get().key);
			if (field != nullptr)
			{
				switch (field->type)
				{
				case BindingType::Bool:
					fieldNode >> *field->Get<bool>(ptr);
					break;
				case BindingType::Int:
					fieldNode >> *field->Get<int>(ptr);
					break;
				case BindingType::Uint:
					fieldNode >> *field->Get<unsigned int>(ptr);
					break;
				case BindingType::Float:
					fieldNode >> *field->Get<float>(ptr);
					break;
				case BindingType::String:
					fieldNode >> *field->Get<String>(ptr);
					break;
				case BindingType::ByteData:
				{
					ByteData data;
					DataWrapper<ByteData> wrapper = { data };
					fieldNode >> wrapper;
					*field->Get<ByteData>(ptr) = std::move(data);
				}
				break;
				case BindingType::IntList:
				{
					List<int> data;
					DataWrapper<List<int>> wrapper = { data };
					fieldNode >> wrapper;
					*field->Get<List<int>>(ptr) = std::move(data);
				}
				break;
				case BindingType::FloatList:
				{
					List<float> data;
					DataWrapper<List<float>> wrapper = { data };
					fieldNode >> wrapper;
					*field->Get<List<float>>(ptr) = std::move(data);
				}
				break;
				case BindingType::StringList:
				{
					List<String>* arrayPointer = field->Get<List<String>>(ptr);
					arrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						String stringValue;
						child >> stringValue;
						arrayPointer->push_back(stringValue);
					}
				}
				break;
				case BindingType::Enum:
					fieldNode >> *field->Get<int>(ptr);
					break;
				case BindingType::Vector2:
					fieldNode >> *field->Get<Vector2>(ptr);
					break;
				case BindingType::Vector2Int:
					fieldNode >> *field->Get<Vector2Int>(ptr);
					break;
				case BindingType::Vector3:
					fieldNode >> *field->Get<Vector3>(ptr);
					break;
				case BindingType::Vector3Int:
					fieldNode >> *field->Get<Vector3Int>(ptr);
					break;
				case BindingType::Vector4:
					fieldNode >> *field->Get<Vector4>(ptr);
					break;
				case BindingType::Vector4Int:
					fieldNode >> *field->Get<Vector4Int>(ptr);
					break;
				case BindingType::Quaternion:
					fieldNode >> *field->Get<Quaternion>(ptr);
					break;
				case BindingType::Color:
					fieldNode >> *field->Get<Color>(ptr);
					break;
				case BindingType::AABB:
					fieldNode >> *field->Get<AABB>(ptr);
					break;
				case BindingType::Matrix:
					fieldNode >> *field->Get<Matrix>(ptr);
					break;
				case BindingType::Vector2List:
				{
					List<Vector2>* arrayPointer = field->Get<List<Vector2>>(ptr);
					arrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						Vector2 data;
						child >> data;
						arrayPointer->push_back(data);
					}
				}
				break;
				case BindingType::Vector3List:
				{
					List<Vector3>* arrayPointer = field->Get<List<Vector3>>(ptr);
					arrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						Vector3 data;
						child >> data;
						arrayPointer->push_back(data);
					}
				}
				break;
				case BindingType::Vector4List:
				{
					List<Vector4>* arrayPointer = field->Get<List<Vector4>>(ptr);
					arrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						Vector4 data;
						child >> data;
						arrayPointer->push_back(data);
					}
				}
				break;
				case BindingType::QuaternionList:
				{
					List<Quaternion>* arrayPointer = field->Get<List<Quaternion>>(ptr);
					arrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						Quaternion data;
						child >> data;
						arrayPointer->push_back(data);
					}
				}
				break;
				case BindingType::MatrixList:
				{
					List<Matrix>* arrayPointer = field->Get<List<Matrix>>(ptr);
					arrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						Matrix data;
						child >> data;
						arrayPointer->push_back(data);
					}
				}
				break;
				case BindingType::Raw:
				{
					ByteData byteData;
					DataWrapper<ByteData> wrapper = { byteData };
					fieldNode >> wrapper;
					memcpy(field->Get<uint8_t>(ptr), byteData.data(), byteData.size());
				}
				break;
				case BindingType::ObjectPtr:
				{
					ObjectPtrData data = {};
					fieldNode >> data;
					Object* obj = GetPtrObject(data);
					*field->Get<ObjectPtr<Object>>(ptr) = obj;
				}
				break;
				case BindingType::ObjectPtrList:
				{
					List<ObjectPtr<Object>>* refArrayPointer = field->Get<List<ObjectPtr<Object>>>(ptr);
					refArrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						ObjectPtrData data = {};
						child >> data;
						refArrayPointer->push_back(GetPtrObject(data));
					}
				}
				break;
				case BindingType::Data:
				{
					Data* data = field->Get<Data>(ptr);
					Context context = Context::CreateNoOffset(data, *field->options.objectType);
					if (context.info != nullptr)
					{
						DeserializeNode(fieldNode, context);
					}
					else
					{
						BB_ERROR("Data class not exists.");
					}
				}
				break;
				case BindingType::DataList:
				{
					ListBase* dataArrayPointer = field->Get<ListBase>(ptr);
					dataArrayPointer->clear_base();
					for (auto& child : fieldNode.GetChildren())
					{
						void* data = dataArrayPointer->emplace_back_base();
						Context context = Context::CreateNoOffset(data, *field->options.objectType);
						if (context.info != nullptr)
						{
							DeserializeNode(child, context);
						}
						else
						{
							BB_ERROR("Data class not exists.");
						}
					}
				}
				break;
				case BindingType::Variant:
				{
					SerializationNodeConstRef typeNode = fieldNode["type"];
					SerializationNodeConstRef dataNode = fieldNode["data"];

					if (typeNode.IsValid() && dataNode.IsValid())
					{
						Variant* variant = field->Get<Variant>(ptr);
						size_t index;
						typeNode >> index;

						switch (index)
						{
						case 0:
						{
							bool value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 1:
						{
							int32_t value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 2:
						{
							uint32_t value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 3:
						{
							int64_t value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 4:
						{
							uint64_t value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 5:
						{
							float value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 6:
						{
							String value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 7:
						{
							Vector2 value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 8:
						{
							Vector2Int value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 9:
						{
							Vector3 value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 10:
						{
							Vector3Int value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 11:
						{
							Vector4 value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 12:
						{
							Vector4Int value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 13:
						{
							Quaternion value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 14:
						{
							Color value;
							dataNode >> value;
							*variant = value;
						}
						break;
						case 15:
						{
							ObjectPtrData data = {};
							dataNode >> data;
							Object* obj = GetPtrObject(data);
							*variant = ObjectPtr<Object>(obj);
						}
						break;
						default:
							BB_INFO("Can't deserialize field " << field->name);
							break;
						}
					}
				}
				break;
				default:
					BB_INFO("Can't deserialize field " << field->name);
					continue;
				}
			}
		}
	}

	void Serializer::AddAdditionalObject(ObjectId objectId)
	{
		auto it = std::find(m_ObjectsToSerialize.begin(), m_ObjectsToSerialize.end(), objectId);
		if (it != m_ObjectsToSerialize.end())
		{
			return;
		}

		FileId fileId = GetFileId(objectId);
		if (m_AdditionalObjectsFileIds.count(fileId) == 0)
		{
			m_AdditionalObjectsFileIds.insert(fileId);
			m_FileIdToObjectId.insert_or_assign(fileId, objectId);
			m_AdditionalObjectsToSerialize.push_back(objectId);
		}
	}

	void Serializer::AddDeserializedObject(ObjectId objectId, FileId fileId)
	{
		ObjectDB::AllocateIdToFileId(objectId, fileId);
		m_FileIdToObjectId.insert_or_assign(fileId, objectId);
		m_DeserializedObjects.push_back(std::make_pair(objectId, fileId));
	}
}
