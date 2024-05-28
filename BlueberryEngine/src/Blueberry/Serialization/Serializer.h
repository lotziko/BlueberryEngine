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

		std::vector<std::pair<Object*, FileId>>& GetDeserializedObjects();

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
		std::vector<Object*> m_ObjectsToSerialize;
		std::vector<Object*> m_AdditionalObjectsToSerialize;
		std::vector<std::pair<Object*, FileId>> m_DeserializedObjects;
		std::unordered_map<FileId, Object*> m_FileIdToObject;
		std::set<FileId> m_AdditionalObjectsIds;
	};
}