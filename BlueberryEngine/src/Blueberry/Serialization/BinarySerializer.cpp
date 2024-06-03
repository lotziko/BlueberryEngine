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
			input.read((char*)&objectCount, sizeof(size_t));
			for (size_t i = 0; i < objectCount; ++i)
			{
				FileId fileId;
				input.read((char*)&fileId, sizeof(FileId));

				size_t typeNameSize;
				input.read((char*)&typeNameSize, sizeof(size_t));

				std::string typeName(typeNameSize, ' ');
				input.read(typeName.data(), typeNameSize);

				auto it = m_FileIdToObject.find(fileId);
				if (it == m_FileIdToObject.end())
				{
					ClassDB::ClassInfo info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
					Object* instance = (Object*)info.createInstance();
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
				input.read((char*)&fileId, sizeof(FileId));
				Object* object = m_FileIdToObject[fileId];
				DeserializeNode(input, Context::Create(object, object->GetType()));
				object->OnCreate();
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
			case BindingType::Int:
				output.write((char*)value.Get<int>(), sizeof(int));
				break;
			case BindingType::Float:
				output.write((char*)value.Get<float>(), sizeof(float));
				break;
			case BindingType::String:
			{
				std::string data = *value.Get<std::string>();
				size_t stringSize = data.size();
				output.write((char*)&stringSize, sizeof(size_t));
				output.write(data.data(), stringSize);
			}
			break;
			case BindingType::ByteData:
			{
				ByteData data = *value.Get<ByteData>();
				output.write((char*)&data.size, sizeof(size_t));
				output.write((char*)data.data, data.size);
			}
			break;
			case BindingType::IntByteArray:
			{
				std::vector<int> data = *value.Get<std::vector<int>>();
				size_t dataSize = data.size();
				output.write((char*)&dataSize, sizeof(size_t));
				output.write((char*)data.data(), data.size() * sizeof(int));
			}
			break;
			case BindingType::FloatByteArray:
			{
				std::vector<float> data = *value.Get<std::vector<float>>();
				size_t dataSize = data.size();
				output.write((char*)&dataSize, sizeof(size_t));
				output.write((char*)data.data(), data.size() * sizeof(float));
			}
			break;
			case BindingType::Enum:
				output.write((char*)value.Get<int>(), sizeof(int));
				break;
			case BindingType::Vector3:
				output.write((char*)&(*value.Get<Vector3>()).x, 3 * sizeof(float));
				break;
			case BindingType::Vector4:
				output.write((char*)&(*value.Get<Vector4>()).x, 4 * sizeof(float));
				break;
			case BindingType::Quaternion:
				output.write((char*)&(*value.Get<Quaternion>()).x, 4 * sizeof(float));
				break;
			case BindingType::Color:
				output.write((char*)&(*value.Get<Color>()).x, 4 * sizeof(float));
				break;
			case BindingType::AABB:
				output.write((char*)&(*value.Get<AABB>()).Center, 6 * sizeof(float));
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
				output.write((char*)&data, sizeof(ObjectPtrData));
			}
			break;
			case BindingType::ObjectPtrArray:
			{
				std::vector<ObjectPtr<Object>> arrayValue = *value.Get<std::vector<ObjectPtr<Object>>>();
				size_t dataSize = arrayValue.size();
				output.write((char*)&dataSize, sizeof(size_t));
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
						output.write((char*)&data, sizeof(ObjectPtrData));
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
			case BindingType::DataArray:
			{
				std::vector<DataPtr<Data>>* arrayValue = value.Get<std::vector<DataPtr<Data>>>();
				size_t dataSize = arrayValue->size();
				output.write((char*)&dataSize, sizeof(size_t));
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
		input.read((char*)&fieldCount, sizeof(size_t));

		auto fieldsMap = context.info.fieldsMap;
		for (size_t i = 0; i < fieldCount; ++i)
		{
			size_t fieldNameSize;
			input.read((char*)&fieldNameSize, sizeof(size_t));
			std::string fieldName(fieldNameSize, ' ');
			input.read(fieldName.data(), fieldNameSize);

			auto fieldIt = fieldsMap.find(fieldName);
			if (fieldIt != fieldsMap.end())
			{
				auto& field = fieldIt->second;
				field.bind->Get(context.ptr, value);

				switch (field.type)
				{
				case BindingType::Int:
					input.read((char*)value.Get<int>(), sizeof(int));
					break;
				case BindingType::Float:
					input.read((char*)value.Get<float>(), sizeof(float));
					break;
				case BindingType::String:
				{
					size_t stringSize;
					input.read((char*)&stringSize, sizeof(size_t));
					std::string data(stringSize, ' ');
					input.read(data.data(), stringSize);
					*value.Get<std::string>() = data;
				}
				break;
				case BindingType::ByteData:
				{
					ByteData data;
					size_t dataSize;
					input.read((char*)&dataSize, sizeof(size_t));
					data.data = new byte[dataSize];
					input.read((char*)data.data, dataSize);
					*value.Get<ByteData>() = data;
				}
				break;
				case BindingType::IntByteArray:
				{
					size_t dataSize;
					input.read((char*)&dataSize, sizeof(size_t));
					std::vector<int> data(dataSize);
					input.read((char*)data.data(), dataSize * sizeof(int));
					*value.Get<std::vector<int>>() = data;
				}
				break;
				case BindingType::FloatByteArray:
				{
					size_t dataSize;
					input.read((char*)&dataSize, sizeof(size_t));
					std::vector<float> data(dataSize);
					input.read((char*)data.data(), dataSize * sizeof(float));
					*value.Get<std::vector<float>>() = data;
				}
				break;
				case BindingType::Enum:
					input.read((char*)value.Get<int>(), sizeof(int));
					break;
				case BindingType::Vector3:
					input.read((char*)&(*value.Get<Vector3>()).x, 3 * sizeof(float));
					break;
				case BindingType::Vector4:
					input.read((char*)&(*value.Get<Vector3>()).x, 4 * sizeof(float));
					break;
				case BindingType::Quaternion:
					input.read((char*)&(*value.Get<Vector3>()).x, 4 * sizeof(float));
					break;
				case BindingType::Color:
					input.read((char*)&(*value.Get<Vector3>()).x, 4 * sizeof(float));
					break;
				case BindingType::AABB:
					input.read((char*)&(*value.Get<AABB>()).Center, 6 * sizeof(float));
					break;
				case BindingType::ObjectPtr:
				{
					ObjectPtrData data = {};
					input.read((char*)&data, sizeof(ObjectPtrData));
					*value.Get<ObjectPtr<Object>>() = GetPtrObject(data);
				}
				break;
				case BindingType::ObjectPtrArray:
				{
					size_t dataSize;
					input.read((char*)&dataSize, sizeof(size_t));
					std::vector<ObjectPtr<Object>>* refArrayPointer = value.Get<std::vector<ObjectPtr<Object>>>();
					for (size_t i = 0; i < dataSize; ++i)
					{
						ObjectPtrData data = {};
						input.read((char*)&data, sizeof(ObjectPtrData));
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
				case BindingType::DataArray:
				{
					size_t dataSize;
					input.read((char*)&dataSize, sizeof(size_t));
					ClassDB::ClassInfo info = ClassDB::GetInfo(field.objectType);
					std::vector<DataPtr<Data>>* dataArrayPointer = value.Get<std::vector<DataPtr<Data>>>();
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
