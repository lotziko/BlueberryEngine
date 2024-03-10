#include "bbpch.h"
#include "ObjectDB.h"

namespace Blueberry
{
	ChunkedObjectArray ObjectDB::s_Array = ChunkedObjectArray();
	std::map<ObjectId, std::pair<Guid, FileId>> ObjectDB::s_ObjectIdToGuid = std::map<ObjectId, std::pair<Guid, FileId>>();
	std::map<Guid, std::map<FileId, ObjectId>> ObjectDB::s_GuidToObjectId = std::map<Guid, std::map<FileId, ObjectId>>();

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

	Object* ObjectDB::GetObject(const ObjectId& id)
	{
		ObjectItem* item = s_Array.GetObjectItem(id);
		return item == nullptr ? nullptr : item->object;
	}

	void ObjectDB::AllocateIdToGuid(const ObjectId& id, const Guid& guid, const FileId& fileId)
	{
		auto s = s_ObjectIdToGuid;
		s_ObjectIdToGuid.insert_or_assign(id, std::pair { guid, fileId });
		s_GuidToObjectId[guid].insert_or_assign(fileId, id);
	}

	void ObjectDB::AllocateIdToGuid(Object* object, const Guid& guid, const FileId& fileId)
	{
		ObjectId objectId = object->GetObjectId();
		AllocateIdToGuid(objectId, guid, fileId);
	}

	void ObjectDB::AllocateEmptyObjectWithGuid(const Guid& guid, const FileId& fileId)
	{
		// TODO do something better
		Object* object = new Object();
		object->SetName("Missing");
		object->SetValid(false);
		ObjectDB::AllocateId(object);
		AllocateIdToGuid(object->GetObjectId(), guid, fileId);
	}

	Guid ObjectDB::GetGuidFromObject(Object* object)
	{
		auto guidIt = s_ObjectIdToGuid.find(object->GetObjectId());
		if (guidIt != s_ObjectIdToGuid.end())
		{
			return guidIt->second.first;
		}
		return Guid();
	}

	std::pair<Guid, FileId> ObjectDB::GetGuidAndFileIdFromObject(Object* object)
	{
		auto guidIt = s_ObjectIdToGuid.find(object->GetObjectId());
		if (guidIt != s_ObjectIdToGuid.end())
		{
			return guidIt->second;
		}
		return std::pair<Guid, FileId>();
	}

	bool ObjectDB::HasGuid(Object* object)
	{
		return s_ObjectIdToGuid.count(object->GetObjectId()) > 0;
	}

	bool ObjectDB::HasGuid(const Guid& guid)
	{
		return s_GuidToObjectId.count(guid) > 0;
	}

	Object* ObjectDB::GetObjectFromGuid(const Guid& guid, const FileId& fileId)
	{
		auto fileIdIt = s_GuidToObjectId.find(guid);
		if (fileIdIt != s_GuidToObjectId.end())
		{
			auto objectIdIt = fileIdIt->second.find(fileId);
			if (objectIdIt != fileIdIt->second.end())
			{
				return ObjectDB::IdToObjectItem(objectIdIt->second)->object;
			}
		}
		return nullptr;
	}
}
