#pragma once

#include "Guid.h"

namespace Blueberry
{
	class Object;

	struct ObjectItem
	{
		Object* object;
	};

	class ChunkedObjectArray
	{
	public:
		ChunkedObjectArray();
		~ChunkedObjectArray();

		const uint32_t MAX_OBJECTS = 2 * 1024 * 1024;
		const uint32_t ELEMENTS_PER_CHUNK = 64 * 1024;

		int32_t AddSingle();
		int32_t AddRange(const int32_t& count);

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
		static void AllocateId(Object* object);
		static void FreeId(Object* object);
		static bool IsValid(Object* object);
		static ObjectItem* IdToObjectItem(const ObjectId& id);

		static void AssignGuid(Object* object, const Guid& guid);
		static Guid GetGuid(Object* object);
		static bool HasGuid(Object* object);
		static Object* GetObject(const Guid& guid);
	private:
		static ChunkedObjectArray s_Array;
		static std::map<ObjectId, Guid> s_ObjectIdToGuid;
		static std::map<Guid, ObjectId> s_GuidToObjectId;
	};
}