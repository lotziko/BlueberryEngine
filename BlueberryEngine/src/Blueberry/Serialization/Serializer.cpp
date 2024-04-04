#include "bbpch.h"
#include "Serializer.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	void Serializer::AddObject(Object* object)
	{
		m_ObjectToFileId.insert({ object, ++m_MaxId });
		m_ObjectsToSerialize.emplace_back(object);
	}

	std::vector<std::pair<Object*, FileId>>& Serializer::GetDeserializedObjects()
	{
		return m_DeserializedObjects;
	}

	FileId Serializer::GetFileId(Object* object)
	{
		if (ObjectDB::HasGuid(object))
		{
			return ObjectDB::GetGuidAndFileIdFromObject(object).second;
		}

		auto idIt = m_ObjectToFileId.find(object);
		if (idIt != m_ObjectToFileId.end())
		{
			return idIt->second;
		}

		FileId id = ++m_MaxId;
		m_ObjectToFileId.insert({ object, id });
		return id;
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
}
