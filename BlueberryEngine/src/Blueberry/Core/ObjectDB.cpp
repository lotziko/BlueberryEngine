#include "bbpch.h"
#include "ObjectDB.h"

namespace Blueberry
{
	ChunkedObjectArray ObjectDB::s_Array = ChunkedObjectArray();
	std::map<ObjectId, Guid> ObjectDB::s_ObjectIdToGuid = std::map<ObjectId, Guid>();
	std::map<Guid, ObjectId> ObjectDB::s_GuidToObjectId = std::map<Guid, ObjectId>();

	ChunkedObjectArray::ChunkedObjectArray()
	{
		m_MaxChunksCount = MAX_OBJECTS / ELEMENTS_PER_CHUNK + 1;
		m_Objects = new ObjectItem*[m_MaxChunksCount];
	}

	ChunkedObjectArray::~ChunkedObjectArray()
	{
		for (uint32_t i = 0; i < m_ChunksCount; i++)
		{
			delete[] m_Objects[i];
		}
		delete[] m_Objects;
	}

	int32_t ChunkedObjectArray::AddSingle()
	{
		return AddRange(1);
	}

	int32_t ChunkedObjectArray::AddRange(const int32_t& count)
	{
		int32_t result = m_ElementsCount;
		ExpandChunksToIndex(m_ElementsCount + count - 1);
		m_ElementsCount += count;
		return result;
	}

	ObjectItem* ChunkedObjectArray::GetObjectItem(const int32_t& index) const
	{
		const uint32_t chunkIndex = index / ELEMENTS_PER_CHUNK;
		const uint32_t inChunkIndex = index % ELEMENTS_PER_CHUNK;
		ObjectItem* chunk = m_Objects[chunkIndex];
		return chunk + inChunkIndex;
	}

	void ChunkedObjectArray::ExpandChunksToIndex(const int32_t& index)
	{
		uint32_t chunkIndex = index / ELEMENTS_PER_CHUNK;
		while (chunkIndex >= m_ChunksCount)
		{
			ObjectItem** chunk = &m_Objects[m_ChunksCount];
			ObjectItem* newChunk = new ObjectItem[ELEMENTS_PER_CHUNK]();
			*chunk = newChunk;
			++m_ChunksCount;
		}
	}

	void ObjectDB::AllocateId(Object* object)
	{
		ObjectId id = s_Array.AddSingle();
		object->m_ObjectId = id;
		ObjectItem* objectItem = IdToObjectItem(id);
		objectItem->object = object;
	}

	void ObjectDB::FreeId(Object* object)
	{
		ObjectId id = object->m_ObjectId;
		ObjectItem* objectItem = IdToObjectItem(id);
		objectItem->object = nullptr;
	}

	bool ObjectDB::IsValid(Object* object)
	{
		ObjectId id = object->m_ObjectId;
		return s_Array.GetObjectItem(id)->object != nullptr;
	}

	ObjectItem* ObjectDB::IdToObjectItem(const ObjectId& id)
	{
		return s_Array.GetObjectItem(id);
	}

	void ObjectDB::AssignGuid(Object* object, const Guid& guid)
	{
		ObjectId objectId = object->GetObjectId();
		s_ObjectIdToGuid.insert_or_assign(objectId, guid);
		s_GuidToObjectId.insert_or_assign(guid, objectId);
	}

	Guid ObjectDB::GetGuid(Object* object)
	{
		auto guidIt = s_ObjectIdToGuid.find(object->GetObjectId());
		if (guidIt != s_ObjectIdToGuid.end())
		{
			return guidIt->second;
		}
		return Guid();
	}

	bool ObjectDB::HasGuid(Object* object)
	{
		return s_ObjectIdToGuid.count(object->GetObjectId()) > 0;
	}

	Object* ObjectDB::GetObject(const Guid& guid)
	{
		auto objectIdIt = s_GuidToObjectId.find(guid);
		if (objectIdIt != s_GuidToObjectId.end())
		{
			return IdToObjectItem(objectIdIt->second)->object;
		}
		return nullptr;
	}
}
