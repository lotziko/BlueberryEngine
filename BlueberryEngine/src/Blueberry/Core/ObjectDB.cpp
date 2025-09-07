#include "Blueberry\Core\ObjectDB.h"

#include <mutex>

namespace Blueberry
{
	static std::mutex s_AllocationMutex = {};

	ChunkedObjectArray ObjectDB::s_Array = ChunkedObjectArray();
	Dictionary<ObjectId, std::pair<Guid, FileId>> ObjectDB::s_ObjectIdToGuid = {};
	Dictionary<Guid, Dictionary<FileId, ObjectId>> ObjectDB::s_GuidToObjectId = {};
	Dictionary<ObjectId, FileId> ObjectDB::s_ObjectIdToFileId = {};

	ChunkedObjectArray::ChunkedObjectArray()
	{
		m_MaxChunksCount = MAX_OBJECTS / ELEMENTS_PER_CHUNK + 1;
		m_Objects = BB_MALLOC_ARRAY(ObjectItem*, m_MaxChunksCount);
	}

	ChunkedObjectArray::~ChunkedObjectArray()
	{
		for (uint32_t i = 0; i < m_ChunksCount; i++)
		{
			BB_FREE(m_Objects[i]);
		}
		BB_FREE(m_Objects);
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

	const uint32_t& ChunkedObjectArray::GetElementsCount()
	{
		return m_ElementsCount;
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
			ObjectItem* newChunk = BB_MALLOC_ARRAY(ObjectItem, ELEMENTS_PER_CHUNK);
			*chunk = newChunk;
			++m_ChunksCount;
		}
	}

	void ObjectDB::AllocateId(Object* object)
	{
		std::lock_guard<std::mutex> lock(s_AllocationMutex);
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

	void ObjectDB::GetObjects(const size_t& type, List<Object*>& result, SearchObjectType searchType)
	{
		bool notAny = searchType != SearchObjectType::Any;
		bool needGuid = searchType == SearchObjectType::WithGuid;
		for (uint32_t i = 0, n = s_Array.GetElementsCount(); i < n; ++i)
		{
			ObjectItem* item = s_Array.GetObjectItem(i);
			if (item != nullptr)
			{
				Object* object = item->object;
				if (object != nullptr && object->IsClassType(type))
				{
					if (notAny)
					{
						if (HasGuid(object) != needGuid)
						{
							continue;
						}
					}
					result.emplace_back(std::move(object));
				}
			}
		}
	}

	void ObjectDB::AllocateIdToFileId(Object* object, const FileId& fileId)
	{
		s_ObjectIdToFileId.insert_or_assign(object->GetObjectId(), fileId);
	}

	void ObjectDB::AllocateIdToGuid(const ObjectId& id, const Guid& guid, const FileId& fileId)
	{
		s_ObjectIdToGuid.insert_or_assign(id, std::make_pair(guid, fileId));
		s_GuidToObjectId[guid].insert_or_assign(fileId, id);
		s_ObjectIdToFileId.insert_or_assign(id, fileId);
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
		object->m_Name = "Missing";
		object->m_State = ObjectState::Missing;
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

	FileId ObjectDB::GetFileIdFromObject(Object* object)
	{
		auto fileIdIt = s_ObjectIdToFileId.find(object->GetObjectId());
		if (fileIdIt != s_ObjectIdToFileId.end())
		{
			return fileIdIt->second;
		}
		return 0;
	}

	FileId ObjectDB::GetFileIdFromObjectId(const ObjectId& id)
	{
		auto fileIdIt = s_ObjectIdToFileId.find(id);
		if (fileIdIt != s_ObjectIdToFileId.end())
		{
			return fileIdIt->second;
		}
		return 0;
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

	bool ObjectDB::HasFileId(Object* object)
	{
		return s_ObjectIdToFileId.count(object->GetObjectId()) > 0;
	}

	bool ObjectDB::HasGuid(Object* object)
	{
		return s_ObjectIdToGuid.count(object->GetObjectId()) > 0;
	}

	bool ObjectDB::HasGuid(const Guid& guid)
	{
		return s_GuidToObjectId.count(guid) > 0;
	}

	bool ObjectDB::HasGuidAndFileId(const Guid& guid, const FileId& fileId)
	{
		auto fileIdIt = s_GuidToObjectId.find(guid);
		if (fileIdIt != s_GuidToObjectId.end())
		{
			return fileIdIt->second.count(fileId) > 0;
		}
		return false;
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

	const Dictionary<FileId, ObjectId>& ObjectDB::GetObjectsFromGuid(const Guid& guid)
	{
		return s_GuidToObjectId[guid];
	}
}
