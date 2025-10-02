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
		uint32_t m_ChunksCount;
		uint32_t m_MaxChunksCount;
		uint32_t m_ElementsCount;
	};

	class ObjectDB
	{
	public:
		static void Initialize();
		static void Shutdown();

		static void AllocateId(Object* object);
		static void FreeId(Object* object);
		static bool IsValid(Object* object);
		static ObjectItem* IdToObjectItem(const ObjectId& id);
		static Object* GetObject(const ObjectId& id);
		static void GetObjects(const size_t& type, List<Object*>& result, SearchObjectType searchType = SearchObjectType::Any);

		static void AllocateIdToFileId(Object* object, const FileId& fileId);
		static void AllocateIdToGuid(const ObjectId& id, const Guid& guid, const FileId& fileId);
		static void AllocateIdToGuid(Object* object, const Guid& guid, const FileId& fileId);
		static void AllocateEmptyObjectWithGuid(const Guid& guid, const FileId& fileId);
		static Guid GetGuidFromObject(Object* object);
		static FileId GetFileIdFromObject(Object* object);
		static FileId GetFileIdFromObjectId(const ObjectId& id);
		static std::pair<Guid, FileId> GetGuidAndFileIdFromObject(Object* object);
		static bool HasFileId(Object* object);
		static bool HasGuid(Object* object);
		static bool HasGuid(const Guid& guid);
		static bool HasGuidAndFileId(const Guid& guid, const FileId& fileId);
		static Object* GetObjectFromGuid(const Guid& guid, const FileId& fileId);
		static const Dictionary<FileId, ObjectId>& GetObjectsFromGuid(const Guid& guid);

	private:
		static ChunkedObjectArray s_Array;
		static Dictionary<ObjectId, std::pair<Guid, FileId>> s_ObjectIdToGuid;
		static Dictionary<Guid, Dictionary<FileId, ObjectId>> s_GuidToObjectId;
		static Dictionary<ObjectId, FileId> s_ObjectIdToFileId;
	};
}