#pragma once
#include "Blueberry\Core\Structs.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	using FileId = uint64_t;

	class Serializer
	{
	protected:
		struct Context
		{
			void* ptr;
			const ClassDB::ClassInfo& info;

			static Context Create(Object* object, const size_t& type)
			{
				const ClassDB::ClassInfo& info = ClassDB::GetInfo(type);
				return { object - info.offset, info };
			}

			static Context CreateNoOffset(Data* data, const size_t& type)
			{
				const ClassDB::ClassInfo& info = ClassDB::GetInfo(type);
				return { data, info };
			}

			static Context Create(Data* data, const ClassDB::ClassInfo& info)
			{
				return { data - info.offset, info };
			}
		};

	public:
		void AddObject(Object* object);
		void AddObject(Object* object, const FileId& fileId);
		virtual void Serialize(const std::string& path) = 0;
		virtual void Deserialize(const std::string& path) = 0;

		List<std::pair<Object*, FileId>>& GetDeserializedObjects();

	protected:
		void AddAdditionalObject(Object* object);
		void AddDeserializedObject(Object* object, const FileId& fileId);

	protected:
		FileId GetFileId(Object* object);
		Object* GetObjectRef(const FileId& fileId);

		Object* GetNextObjectToSerialize();
		Object* GetPtrObject(const ObjectPtrData& data);
		ObjectPtrData GetPtrData(Object* object);
		FileId GenerateFileId();

	protected:
		Guid m_AssetGuid;
		List<Object*> m_ObjectsToSerialize;
		List<Object*> m_AdditionalObjectsToSerialize;
		List<std::pair<Object*, FileId>> m_DeserializedObjects;
		Dictionary<FileId, Object*> m_FileIdToObject;
		HashSet<FileId> m_AdditionalObjectsIds;
	};
}