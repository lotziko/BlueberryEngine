#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class Object;

	struct ObjectItem
	{
		Object* object;
	};

	enum class SearchObjectType
	{
		Any,
		WithGuid,
		WithoutGuid
	};

	class ChunkedObjectArray;

	struct ObjectIterator
	{
		ObjectIterator(TypeId type, SearchObjectType searchType, ChunkedObjectArray* objectArray, int32_t index) : type(type), searchType(searchType), objectArray(objectArray), index(index)
		{
		}

		ObjectIterator& operator++();
		ObjectIterator& operator--();

		Object* operator*() const;
		Object* operator->() const;

		void FindNext();

		bool operator!= (ObjectIterator other) const;
		bool operator== (ObjectIterator other) const;

		TypeId type = 0;
		SearchObjectType searchType = SearchObjectType::Any;
		ChunkedObjectArray* objectArray;
		int32_t index = 0;
		Object* object = nullptr;
	};

	struct ObjectView
	{
		ObjectView(TypeId type, SearchObjectType searchType, ChunkedObjectArray* objectArray, int32_t index) : type(type), searchType(searchType), objectArray(objectArray), index(index)
		{
		}

		ObjectIterator begin() const;
		ObjectIterator end() const;

		TypeId type = 0;
		SearchObjectType searchType = SearchObjectType::Any;
		ChunkedObjectArray* objectArray;
		int32_t index = 0;
	};

	class ChunkedObjectArray
	{
	public:
		void Initialize();
		void Shutdown();

		const uint32_t MAX_OBJECTS = 2 * 1024 * 1024;
		const uint32_t ELEMENTS_PER_CHUNK = 64 * 1024;

		int32_t AddSingle();
		int32_t AddRange(const int32_t& count);

		const uint32_t& GetElementsCount();

		ObjectItem* GetObjectItem(const int32_t& index) const;

	private:
		void ExpandChunksToIndex(const int32_t& index);

	private:
		ObjectItem** m_Objects;
		uint32_t m_ChunksCount = 0;
		uint32_t m_MaxChunksCount = 0;
		uint32_t m_ElementsCount = 0;
	};

	class BB_API ObjectDB
	{
	public:
		static void Initialize();
		static void Shutdown();

		static void AllocateId(Object* object);
		static void FreeId(Object* object);
		static bool IsValid(Object* object);
		static ObjectItem* IdToObjectItem(ObjectId id);
		static Object* GetObject(ObjectId id);
		static ObjectView GetObjects(TypeId type, SearchObjectType searchType = SearchObjectType::Any);

		static void AllocateIdToFileId(ObjectId id, FileId fileId);
		static void AllocateIdToFileId(Object* object, FileId fileId);
		static void AllocateIdToGuid(ObjectId id, const Guid& guid, FileId fileId);
		static void AllocateIdToGuid(Object* object, const Guid& guid, FileId fileId);
		static void AllocateEmptyObjectWithGuid(const Guid& guid, FileId fileId);
		static Guid GetGuidFromObject(Object* object);
		static Guid GetGuidFromObjectId(ObjectId id);
		static FileId GetFileIdFromObject(Object* object);
		static FileId GetFileIdFromObjectId(ObjectId id);
		static std::pair<Guid, FileId> GetGuidAndFileIdFromObject(ObjectId id);
		static std::pair<Guid, FileId> GetGuidAndFileIdFromObject(Object* object);
		static bool HasFileId(ObjectId id);
		static bool HasFileId(Object* object);
		static bool HasGuid(ObjectId id);
		static bool HasGuid(Object* object);
		static bool HasGuid(const Guid& guid);
		static bool HasGuidAndFileId(const Guid& guid, FileId fileId);
		static Object* GetObjectFromGuid(const Guid& guid, FileId fileId);
		static const Dictionary<FileId, ObjectId>& GetObjectsFromGuid(const Guid& guid);

	private:
		static ChunkedObjectArray s_Array;
		static Dictionary<ObjectId, std::pair<Guid, FileId>> s_ObjectIdToGuid;
		static Dictionary<Guid, Dictionary<FileId, ObjectId>> s_GuidToObjectId;
		static Dictionary<ObjectId, FileId> s_ObjectIdToFileId;
	};
}