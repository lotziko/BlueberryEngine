#include "bbpch.h"
#include "BinarySerializer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"
#include <fstream>

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	void BinarySerializer::Serialize(const std::string& path)
	{
		m_AssetGuid = ObjectDB::GetGuidFromObject(m_ObjectsToSerialize[0]);
		std::ofstream output(path, std::ofstream::binary);
		if (output.is_open())
		{
			size_t objectCount = 0;
			std::stringstream headers;
			std::stringstream objects;

			Object* object;
			while ((object = GetNextObjectToSerialize()) != nullptr)
			{
				FileId fileId = GetFileId(object);
				headers.write((char*)&fileId, sizeof(FileId));

				std::string typeName = object->GetTypeName();
				size_t typeNameSize = typeName.size();
				headers.write((char*)&typeNameSize, sizeof(size_t));
				headers.write(typeName.c_str(), typeNameSize);

				objects.write((char*)&fileId, sizeof(FileId));
				SerializeNode(objects, Context::Create(object, object->GetType()));
				++objectCount;
			}

			output.write((char*)&objectCount, sizeof(size_t));
			output << headers.rdbuf();
			output << objects.rdbuf();
			output.close();
		}
	}

	void BinarySerializer::Deserialize(const std::string& path)
	{
		std::ifstream input(path, std::ifstream::binary);
		if (input.is_open())
		{
			size_t objectCount;
			input.read(reinterpret_cast<char*>(&objectCount), sizeof(size_t));
			for (size_t i = 0; i < objectCount; ++i)
			{
				FileId fileId;
				input.read(reinterpret_cast<char*>(&fileId), sizeof(FileId));

				size_t typeNameSize;
				input.read(reinterpret_cast<char*>(&typeNameSize), sizeof(size_t));

				std::string typeName(typeNameSize, ' ');
				input.read(typeName.data(), typeNameSize);

				auto it = m_FileIdToObject.find(fileId);
				if (it == m_FileIdToObject.end())
				{
					ClassDB::ClassInfo info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
					Object* instance = info.createInstance();
					AddDeserializedObject(instance, fileId);
				}
				else
				{
					Object* instance = it->second;
					AddDeserializedObject(instance, fileId);
				}
			}

			for (size_t i = 0; i < objectCount; ++i)
			{
				FileId fileId;
				input.read(reinterpret_cast<char*>(&fileId), sizeof(FileId));
				Object* object = m_FileIdToObject[fileId];
				DeserializeNode(input, Context::Create(object, object->GetType()));
			}
			input.close();
		}
	}

	void BinarySerializer::SerializeNode(std::stringstream& output, Context context)
	{
		Variant value;
		auto fields = context.info.fields;
		size_t fieldCount = fields.size();
		output.write((char*)&fieldCount, sizeof(size_t));

		for (auto& field : fields)
		{
			field.bind->Get(context.ptr, value);
			size_t fieldNameSize = field.name.size();
			output.write((char*)&fieldNameSize, sizeof(size_t));
			output.write(field.name.c_str(), fieldNameSize);

			switch (field.type)
			{
			case BindingType::Bool:
				output.write(reinterpret_cast<char*>(value.Get<bool>()), sizeof(bool));
				break;
			case BindingType::Int:
				output.write(reinterpret_cast<char*>(value.Get<int>()), sizeof(int));
				break;
			case BindingType::Float:
				output.write(reinterpret_cast<char*>(value.Get<float>()), sizeof(float));
				break;
			case BindingType::String:
			{
				std::string data = *value.Get<std::string>();
				size_t stringSize = data.size();
				output.write(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
				output.write(data.data(), stringSize);
			}
			break;
			case BindingType::ByteData:
			{
				ByteData data = *value.Get<ByteData>();
				output.write(reinterpret_cast<char*>(&data.size), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data), data.size);
			}
			break;
			case BindingType::IntList:
			{
				List<int> data = *value.Get<List<int>>();
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(int));
			}
			break;
			case BindingType::FloatList:
			{
				List<float> data = *value.Get<List<float>>();
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
			}
			break;
			case BindingType::StringList:
			{
				List<std::string> data = *value.Get<List<std::string>>();
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				for (size_t i = 0; i < dataSize; ++i)
				{
					std::string string = data[i];
					size_t stringSize = string.size();
					output.write(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
					output.write(string.data(), stringSize);
				}
			}
			break;
			case BindingType::Enum:
				output.write(reinterpret_cast<char*>(value.Get<int>()), sizeof(int));
				break;
			case BindingType::Vector3:
				output.write(reinterpret_cast<char*>(&(*value.Get<Vector3>()).x), 3 * sizeof(float));
				break;
			case BindingType::Vector4:
				output.write(reinterpret_cast<char*>(&(*value.Get<Vector4>()).x), 4 * sizeof(float));
				break;
			case BindingType::Quaternion:
				output.write(reinterpret_cast<char*>(&(*value.Get<Quaternion>()).x), 4 * sizeof(float));
				break;
			case BindingType::Color:
				output.write(reinterpret_cast<char*>(&(*value.Get<Color>()).x), 4 * sizeof(float));
				break;
			case BindingType::AABB:
				output.write(reinterpret_cast<char*>(&(*value.Get<AABB>()).Center), 6 * sizeof(float));
				break;
			case BindingType::Raw:
				output.write(reinterpret_cast<char*>(value.Get()), field.objectType);
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
				output.write(reinterpret_cast<char*>(&data), sizeof(ObjectPtrData));
			}
			break;
			case BindingType::ObjectPtrArray:
			{
				List<ObjectPtr<Object>> arrayValue = *value.Get<List<ObjectPtr<Object>>>();
				size_t dataSize = arrayValue.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				if (dataSize > 0)
				{
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
						output.write(reinterpret_cast<char*>(&data), sizeof(ObjectPtrData));
					}
				}
			}
			break;
			case BindingType::Data:
			{
				DataPtr<Data> dataValue = *value.Get<DataPtr<Data>>();
				Data* data = dataValue.Get();
				Context context = Context::CreateNoOffset(data, field.objectType);
				SerializeNode(output, context);
			}
			break;
			case BindingType::DataList:
			{
				List<DataPtr<Data>>* arrayValue = value.Get<List<DataPtr<Data>>>();
				size_t dataSize = arrayValue->size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				if (dataSize > 0)
				{
					for (auto const& dataValue : *arrayValue)
					{
						Data* data = dataValue.Get();
						Context context = Context::CreateNoOffset(data, field.objectType);
						SerializeNode(output, context);
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

	void BinarySerializer::DeserializeNode(std::ifstream& input, Context context)
	{
		Variant value;
		size_t fieldCount;
		input.read(reinterpret_cast<char*>(&fieldCount), sizeof(size_t));

		auto fieldsMap = context.info.fieldsMap;
		for (size_t i = 0; i < fieldCount; ++i)
		{
			size_t fieldNameSize;
			input.read(reinterpret_cast<char*>(&fieldNameSize), sizeof(size_t));
			std::string fieldName(fieldNameSize, ' ');
			input.read(fieldName.data(), fieldNameSize);

			auto fieldIt = fieldsMap.find(fieldName);
			if (fieldIt != fieldsMap.end())
			{
				auto& field = fieldIt->second;
				field.bind->Get(context.ptr, value);

				switch (field.type)
				{
				case BindingType::Bool:
					input.read(reinterpret_cast<char*>(value.Get<bool>()), sizeof(bool));
					break;
				case BindingType::Int:
					input.read(reinterpret_cast<char*>(value.Get<int>()), sizeof(int));
					break;
				case BindingType::Float:
					input.read(reinterpret_cast<char*>(value.Get<float>()), sizeof(float));
					break;
				case BindingType::String:
				{
					size_t stringSize;
					input.read(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
					std::string data(stringSize, ' ');
					input.read(data.data(), stringSize);
					*value.Get<std::string>() = data;
				}
				break;
				case BindingType::ByteData:
				{
					ByteData data;
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					data.data = BB_MALLOC_ARRAY(uint8_t, dataSize);
					input.read(reinterpret_cast<char*>(data.data), dataSize);
					*value.Get<ByteData>() = data;
				}
				break;
				case BindingType::IntList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<int> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(int));
					*value.Get<List<int>>() = data;
				}
				break;
				case BindingType::FloatList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<float> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(float));
					*value.Get<List<float>>() = data;
				}
				break;
				case BindingType::StringList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<std::string> data(dataSize);
					for (size_t i = 0; i < dataSize; ++i)
					{
						size_t stringSize;
						input.read(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
						std::string string(stringSize, ' ');
						input.read(string.data(), stringSize);
						data[i] = string;
					}
					*value.Get<List<std::string>>() = data;
				}
				break;
				case BindingType::Enum:
					input.read(reinterpret_cast<char*>(value.Get<int>()), sizeof(int));
					break;
				case BindingType::Vector3:
					input.read(reinterpret_cast<char*>(&(*value.Get<Vector3>()).x), 3 * sizeof(float));
					break;
				case BindingType::Vector4:
					input.read(reinterpret_cast<char*>(&(*value.Get<Vector3>()).x), 4 * sizeof(float));
					break;
				case BindingType::Quaternion:
					input.read(reinterpret_cast<char*>(&(*value.Get<Vector3>()).x), 4 * sizeof(float));
					break;
				case BindingType::Color:
					input.read(reinterpret_cast<char*>(&(*value.Get<Vector3>()).x), 4 * sizeof(float));
					break;
				case BindingType::AABB:
					input.read(reinterpret_cast<char*>(&(*value.Get<AABB>()).Center), 6 * sizeof(float));
					break;
				case BindingType::Raw:
				{
					char* data = reinterpret_cast<char*>(value.Get());
					input.read(data, field.objectType);
				}
				break;
				case BindingType::ObjectPtr:
				{
					ObjectPtrData data = {};
					input.read(reinterpret_cast<char*>(&data), sizeof(ObjectPtrData));
					*value.Get<ObjectPtr<Object>>() = GetPtrObject(data);
				}
				break;
				case BindingType::ObjectPtrArray:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<ObjectPtr<Object>>* refArrayPointer = value.Get<List<ObjectPtr<Object>>>();
					for (size_t i = 0; i < dataSize; ++i)
					{
						ObjectPtrData data = {};
						input.read(reinterpret_cast<char*>(&data), sizeof(ObjectPtrData));
						refArrayPointer->emplace_back(GetPtrObject(data));
					}
				}
				break;
				case BindingType::Data:
				{
					const ClassDB::ClassInfo& info = ClassDB::GetInfo(field.objectType);
					Data* instance = info.createDataInstance();
					Context context = Context::Create(instance, info);
					DeserializeNode(input, context);
					*value.Get<DataPtr<Data>>() = context.ptr;
				}
				break;
				case BindingType::DataList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					ClassDB::ClassInfo info = ClassDB::GetInfo(field.objectType);
					List<DataPtr<Data>>* dataArrayPointer = value.Get<List<DataPtr<Data>>>();
					for (size_t i = 0; i < dataSize; ++i)
					{
						Data* instance = info.createDataInstance();
						Context context = Context::Create(instance, info);
						DeserializeNode(input, context);
						dataArrayPointer->emplace_back(context.ptr);
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
