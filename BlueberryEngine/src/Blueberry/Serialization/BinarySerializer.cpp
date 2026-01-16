#include "Blueberry\Serialization\BinarySerializer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"

#include <fstream>
#include <sstream>

namespace Blueberry
{
	void BinarySerializer::Serialize(const String& path)
	{
		m_AssetGuid = ObjectDB::GetGuidFromObject(m_ObjectsToSerialize[0]);
		std::ofstream output(path.data(), std::ofstream::binary);
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

				String typeName = object->GetTypeName();
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

	void BinarySerializer::Deserialize(const String& path)
	{
		std::ifstream input(path.data(), std::ifstream::binary);
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

				String typeName(typeNameSize, ' ');
				input.read(typeName.data(), typeNameSize);

				auto it = m_FileIdToObject.find(fileId);
				if (it == m_FileIdToObject.end())
				{
					const ClassInfo* info = ClassDB::GetInfo(TO_OBJECT_TYPE(typeName));
					if (info == nullptr)
					{
						BB_ERROR("Class not exists.");
						continue;
					}
					Object* instance = info->createInstance();
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
				if (object != nullptr)
				{
					DeserializeNode(input, Context::Create(object, object->GetType()));
				}
			}
			input.close();
		}
		if (m_FinalizeObjectCallback != nullptr)
		{
			for (size_t i = 0; i < m_DeserializedObjects.size(); ++i)
			{
				auto& pair = m_DeserializedObjects[i];
				m_FinalizeObjectCallback(pair.first, m_AssetGuid, pair.second);
			}
		}
	}

	void BinarySerializer::SerializeNode(std::stringstream& output, Context context)
	{
		size_t fieldCount = context.info->fields.size();
		output.write((char*)&fieldCount, sizeof(size_t));

		for (auto& field : context.info->fields)
		{
			void* ptr = context.ptr;

			size_t fieldNameSize = field.name.size();
			output.write((char*)&fieldNameSize, sizeof(size_t));
			output.write(field.name.c_str(), fieldNameSize);

			switch (field.type)
			{
			case BindingType::Bool:
				output.write(reinterpret_cast<char*>(field.Get<bool>(ptr)), sizeof(bool));
				break;
			case BindingType::Int:
				output.write(reinterpret_cast<char*>(field.Get<int>(ptr)), sizeof(int));
				break;
			case BindingType::Uint:
				output.write(reinterpret_cast<char*>(field.Get<unsigned int>(ptr)), sizeof(unsigned int));
				break;
			case BindingType::Float:
				output.write(reinterpret_cast<char*>(field.Get<float>(ptr)), sizeof(float));
				break;
			case BindingType::String:
			{
				String data = *field.Get<String>(ptr);
				size_t stringSize = data.size();
				output.write(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
				output.write(data.data(), stringSize);
			}
			break;
			case BindingType::ByteData:
			{
				ByteData data = *field.Get<ByteData>(ptr);
				size_t size = data.size();
				output.write(reinterpret_cast<char*>(&size), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size());
			}
			break;
			case BindingType::IntList:
			{
				List<int> data = *field.Get<List<int>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(int));
			}
			break;
			case BindingType::FloatList:
			{
				List<float> data = *field.Get<List<float>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
			}
			break;
			case BindingType::StringList:
			{
				List<String> data = *field.Get<List<String>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				for (size_t i = 0; i < dataSize; ++i)
				{
					String string = data[i];
					size_t stringSize = string.size();
					output.write(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
					output.write(string.data(), stringSize);
				}
			}
			break;
			case BindingType::Enum:
				output.write(reinterpret_cast<char*>(field.Get<int>(ptr)), sizeof(int));
				break;
			case BindingType::Vector2:
				output.write(reinterpret_cast<char*>(field.Get<Vector2>(ptr)), sizeof(Vector2));
				break;
			case BindingType::Vector3:
				output.write(reinterpret_cast<char*>(field.Get<Vector3>(ptr)), sizeof(Vector3));
				break;
			case BindingType::Vector4:
				output.write(reinterpret_cast<char*>(field.Get<Vector4>(ptr)), sizeof(Vector4));
				break;
			case BindingType::Quaternion:
				output.write(reinterpret_cast<char*>(field.Get<Quaternion>(ptr)), sizeof(Quaternion));
				break;
			case BindingType::Color:
				output.write(reinterpret_cast<char*>(field.Get<Color>(ptr)), sizeof(Color));
				break;
			case BindingType::AABB:
				output.write(reinterpret_cast<char*>(field.Get<AABB>(ptr)), sizeof(AABB));
				break;
			case BindingType::Matrix:
				output.write(reinterpret_cast<char*>(field.Get<Matrix>(ptr)), sizeof(Matrix));
				break;
			case BindingType::Vector2List:
			{
				List<Vector2> data = *field.Get<List<Vector2>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(Vector2));
			}
			break;
			case BindingType::Vector3List:
			{
				List<Vector3> data = *field.Get<List<Vector3>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(Vector3));
			}
			break;
			case BindingType::Vector4List:
			{
				List<Vector4> data = *field.Get<List<Vector4>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(Vector4));
			}
			break;
			case BindingType::QuaternionList:
			{
				List<Quaternion> data = *field.Get<List<Quaternion>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(Quaternion));
			}
			break;
			case BindingType::MatrixList:
			{
				List<Matrix> data = *field.Get<List<Matrix>>(ptr);
				size_t dataSize = data.size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				output.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(Matrix));
			}
			break;
			case BindingType::Raw:
				output.write(reinterpret_cast<char*>(field.Get<char*>(ptr)), field.options.size);
				break;
			case BindingType::ObjectPtr:
			{
				ObjectPtr<Object>* objectRefValue = field.Get<ObjectPtr<Object>>(ptr);
				ObjectPtrData data = {};
				if (objectRefValue->IsValid())
				{
					data = GetPtrData(objectRefValue->Get());
				}
				else
				{
					data.fileId = 0;
				}
				output.write(reinterpret_cast<char*>(&data), sizeof(ObjectPtrData));
			}
			break;
			case BindingType::ObjectPtrList:
			{
				List<ObjectPtr<Object>>* arrayPointer = field.Get<List<ObjectPtr<Object>>>(ptr);
				size_t dataSize = arrayPointer->size();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				if (dataSize > 0)
				{
					for (ObjectPtr<Object>& objectRefValue : *arrayPointer)
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
				Data* data = field.Get<Data>(ptr);
				Context context = Context::CreateNoOffset(data, field.options.objectType);
				SerializeNode(output, context);
			}
			break;
			case BindingType::DataList:
			{
				ListBase* dataArrayPointer = field.Get<ListBase>(ptr);
				size_t dataSize = dataArrayPointer->size_base();
				output.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
				if (dataSize > 0)
				{
					for (uint32_t i = 0; i < dataSize; ++i)
					{
						void* data = dataArrayPointer->get_base(i);
						Context context = Context::CreateNoOffset(data, field.options.objectType);
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
		size_t fieldCount;
		input.read(reinterpret_cast<char*>(&fieldCount), sizeof(size_t));

		for (size_t i = 0; i < fieldCount; ++i)
		{
			size_t fieldNameSize;
			input.read(reinterpret_cast<char*>(&fieldNameSize), sizeof(size_t));
			String fieldName(fieldNameSize, ' ');
			input.read(fieldName.data(), fieldNameSize);

			auto fieldIt = context.info->fieldsMap.find(fieldName);
			if (fieldIt != context.info->fieldsMap.end())
			{
				auto& field = fieldIt->second;
				void* ptr = context.ptr;

				switch (field.type)
				{
				case BindingType::Bool:
					input.read(reinterpret_cast<char*>(field.Get<bool>(ptr)), sizeof(bool));
					break;
				case BindingType::Int:
					input.read(reinterpret_cast<char*>(field.Get<int>(ptr)), sizeof(int));
					break;
				case BindingType::Uint:
					input.read(reinterpret_cast<char*>(field.Get<unsigned int>(ptr)), sizeof(unsigned int));
					break;
				case BindingType::Float:
					input.read(reinterpret_cast<char*>(field.Get<float>(ptr)), sizeof(float));
					break;
				case BindingType::String:
				{
					size_t stringSize;
					input.read(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
					String data(stringSize, ' ');
					input.read(data.data(), stringSize);
					*field.Get<String>(ptr) = std::move(data);
				}
				break;
				case BindingType::ByteData:
				{
					ByteData data;
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					data.resize(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize);
					*field.Get<ByteData>(ptr) = std::move(data);
				}
				break;
				case BindingType::IntList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<int> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(int));
					*field.Get<List<int>>(ptr) = std::move(data);
				}
				break;
				case BindingType::FloatList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<float> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(float));
					*field.Get<List<float>>(ptr) = std::move(data);
				}
				break;
				case BindingType::StringList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<String> data(dataSize);
					for (size_t i = 0; i < dataSize; ++i)
					{
						size_t stringSize;
						input.read(reinterpret_cast<char*>(&stringSize), sizeof(size_t));
						String string(stringSize, ' ');
						input.read(string.data(), stringSize);
						data[i] = string;
					}
					*field.Get<List<String>>(ptr) = std::move(data);
				}
				break;
				case BindingType::Enum:
					input.read(reinterpret_cast<char*>(field.Get<int>(ptr)), sizeof(int));
					break;
				case BindingType::Vector2:
					input.read(reinterpret_cast<char*>(field.Get<Vector2>(ptr)), sizeof(Vector2));
					break;
				case BindingType::Vector3:
					input.read(reinterpret_cast<char*>(field.Get<Vector3>(ptr)), sizeof(Vector3));
					break;
				case BindingType::Vector4:
					input.read(reinterpret_cast<char*>(field.Get<Vector3>(ptr)), sizeof(Vector4));
					break;
				case BindingType::Quaternion:
					input.read(reinterpret_cast<char*>(field.Get<Vector3>(ptr)), sizeof(Quaternion));
					break;
				case BindingType::Color:
					input.read(reinterpret_cast<char*>(field.Get<Vector3>(ptr)), sizeof(Color));
					break;
				case BindingType::Matrix:
					input.read(reinterpret_cast<char*>(field.Get<Matrix>(ptr)), sizeof(Matrix));
					break;
				case BindingType::AABB:
					input.read(reinterpret_cast<char*>(field.Get<AABB>(ptr)), sizeof(AABB));
					break;
				case BindingType::Vector2List:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<Vector2> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(Vector2));
					*field.Get<List<Vector2>>(ptr) = std::move(data);
				}
				break;
				case BindingType::Vector3List:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<Vector3> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(Vector3));
					*field.Get<List<Vector3>>(ptr) = std::move(data);
				}
				break;
				case BindingType::Vector4List:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<Vector4> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(Vector4));
					*field.Get<List<Vector4>>(ptr) = std::move(data);
				}
				break;
				case BindingType::QuaternionList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<Quaternion> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(Quaternion));
					*field.Get<List<Quaternion>>(ptr) = std::move(data);
				}
				break;
				case BindingType::MatrixList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<Matrix> data(dataSize);
					input.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(Matrix));
					*field.Get<List<Matrix>>(ptr) = std::move(data);
				}
				break;
				case BindingType::Raw:
				{
					char* data = reinterpret_cast<char*>(field.Get<char*>(ptr));
					input.read(data, field.options.size);
				}
				break;
				case BindingType::ObjectPtr:
				{
					ObjectPtrData data = {};
					input.read(reinterpret_cast<char*>(&data), sizeof(ObjectPtrData));
					*field.Get<ObjectPtr<Object>>(ptr) = GetPtrObject(data);
				}
				break;
				case BindingType::ObjectPtrList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					List<ObjectPtr<Object>>* refArrayPointer = field.Get<List<ObjectPtr<Object>>>(ptr);
					refArrayPointer->clear_base();
					for (size_t i = 0; i < dataSize; ++i)
					{
						ObjectPtrData data = {};
						input.read(reinterpret_cast<char*>(&data), sizeof(ObjectPtrData));
						refArrayPointer->push_back(GetPtrObject(data));
					}
				}
				break;
				case BindingType::Data:
				{
					void* data = field.Get<void*>(ptr);
					Context context = Context::CreateNoOffset(data, field.options.objectType);
					DeserializeNode(input, context);
				}
				break;
				case BindingType::DataList:
				{
					size_t dataSize;
					input.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
					ListBase* dataArrayPointer = field.Get<ListBase>(ptr);
					dataArrayPointer->clear_base();
					for (size_t i = 0; i < dataSize; ++i)
					{
						void* data = dataArrayPointer->emplace_back_base();
						Context context = Context::CreateNoOffset(data, field.options.objectType);
						DeserializeNode(input, context);
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
