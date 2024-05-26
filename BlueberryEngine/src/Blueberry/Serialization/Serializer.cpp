#include "bbpch.h"
#include "Serializer.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Assets\AssetLoader.h"

#include <iomanip>
#include <random>

namespace Blueberry
{
	void Serializer::AddObject(Object* object)
	{
		FileId fileId = GetFileId(object);
		m_FileIdToObject.insert({ fileId, object });
		m_ObjectsToSerialize.emplace_back(object);
	}

	void Serializer::AddObject(Object* object, const FileId& fileId)
	{
		m_FileIdToObject.insert({ fileId, object });
	}

	std::vector<std::pair<Object*, FileId>>& Serializer::GetDeserializedObjects()
	{
		return m_DeserializedObjects;
	}

	void Serializer::AddAdditionalObject(Object* object)
	{
		FileId fileId = GetFileId(object);
		if (m_FileIdToObject.count(fileId) == 0)
		{
			m_FileIdToObject.insert_or_assign(fileId, object);
			m_AdditionalObjectsToSerialize.emplace_back(object);
		}
	}

	void Serializer::AddDeserializedObject(Object* object, const FileId& fileId)
	{
		ObjectDB::AllocateIdToFileId(object, fileId);
		m_FileIdToObject.insert_or_assign(fileId, object);
		m_DeserializedObjects.emplace_back(std::make_pair(object, fileId));
	}

	FileId Serializer::GetFileId(Object* object)
	{
		if (ObjectDB::HasFileId(object))
		{
			return ObjectDB::GetFileIdFromObject(object);
		}

		FileId fileId = GenerateFileId();
		m_FileIdToObject.insert_or_assign(fileId, object);
		ObjectDB::AllocateIdToFileId(object, fileId);
		return fileId;
	}

	Object* Serializer::GetObjectRef(const FileId& fileId)
	{
		auto idIt = m_FileIdToObject.find(fileId);
		if (idIt != m_FileIdToObject.end())
		{
			return idIt->second;
		}
		return nullptr;
	}

	Object* Serializer::GetNextObjectToSerialize()
	{
		if (m_ObjectsToSerialize.size() == 0 && m_AdditionalObjectsToSerialize.size() == 0)
		{
			return nullptr;
		}

		Object* object;
		if (m_AdditionalObjectsToSerialize.size() > 0)
		{
			object = m_AdditionalObjectsToSerialize.front();
			m_AdditionalObjectsToSerialize.erase(m_AdditionalObjectsToSerialize.begin());
		}
		else
		{
			object = m_ObjectsToSerialize[0];
			m_ObjectsToSerialize.erase(m_ObjectsToSerialize.begin());
		}
		return object;
	}

	Object* Serializer::GetPtrObject(const ObjectPtrData& data)
	{
		Object* result;
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
		ObjectPtrData result = {};
		if (ObjectDB::HasGuid(object))
		{
			FileId fileId = ObjectDB::GetFileIdFromObject(object);
			auto fileIdIt = m_FileIdToObject.find(fileId);
			if (fileIdIt != m_FileIdToObject.end() && fileIdIt->second == object)
			{
				result.fileId = fileId;
			}
			else
			{
				auto pair = ObjectDB::GetGuidAndFileIdFromObject(object);
				result.guid = pair.first;
				result.fileId = pair.second;
			}
		}
		else
		{
			FileId fileId;
			if (!ObjectDB::HasFileId(object))
			{
				fileId = GetFileId(object);
				AddAdditionalObject(object);
			}
			else
			{
				fileId = ObjectDB::GetFileIdFromObject(object);
				if (m_FileIdToObject.count(fileId) == 0)
				{
					AddAdditionalObject(object);
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

		static std::uniform_int_distribution<unsigned long long> dis(
			(std::numeric_limits<std::uint64_t>::min)(),
			(std::numeric_limits<std::uint64_t>::max)()
		);

		return dis(gen);
	}
}
