#pragma once

#include "Blueberry\Core\Structs.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\SerializationTree.h"

namespace Blueberry
{
	class SerializerBackend;

	class Serializer
	{
	public:
		virtual void Serialize(const String& path, const bool& isText);
		virtual void Deserialize(const String& path);

		const Guid& GetGuid();
		void SetGuid(const Guid& guid);

		List<std::pair<ObjectId, FileId>>& GetDeserializedObjects();

		void AddObject(Object* object);
		void AddObject(Object* object, const FileId& fileId);

	protected:
		struct Context
		{
			void* ptr;
			const ClassInfo* info;

			static Context Create(Object* object, const size_t& type)
			{
				const ClassInfo* info = ClassDB::GetInfo(type);
				return { object - info->offset, info };
			}

			static Context CreateNoOffset(Data* data, const size_t& type)
			{
				const ClassInfo* info = ClassDB::GetInfo(type);
				return { data, info };
			}

			static Context CreateNoOffset(void* data, const size_t& type)
			{
				const ClassInfo* info = ClassDB::GetInfo(type);
				return { data, info };
			}

			static Context Create(Data* data, const ClassInfo* info)
			{
				return { data - info->offset, info };
			}
		};

	protected:
		FileId GetFileId(const ObjectId& objectId);
		Object* GetObjectRef(const FileId& fileId);
		Object* GetNextObjectToSerialize();

		Object* GetPtrObject(const ObjectPtrData& data);
		ObjectPtrData GetPtrData(Object* object);
		FileId GenerateFileId();

		void SerializeNode(SerializationNodeRef node, Context context);
		void DeserializeNode(SerializationNodeConstRef node, Context context);

		virtual void AddAdditionalObject(const ObjectId& objectId);
		void AddDeserializedObject(const ObjectId& objectId, const FileId& fileId);

	protected:
		Dictionary<FileId, ObjectId> m_FileIdToObjectId;
		HashSet<FileId> m_AdditionalObjectsFileIds;
		List<ObjectId> m_ObjectsToSerialize;
		List<ObjectId> m_AdditionalObjectsToSerialize;
		List<std::pair<ObjectId, FileId>> m_DeserializedObjects;
		List<SerializationTree> m_Trees;
		HashSet<ObjectId> m_References;
		Guid m_Guid;
		Object* m_CurrentObject;
	};
}