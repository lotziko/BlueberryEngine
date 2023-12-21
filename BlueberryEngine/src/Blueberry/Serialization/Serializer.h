#pragma once

namespace Blueberry
{
	using FileId = uint64_t;

	class Serializer
	{
	public:
		void AddObject(Object* object);
		virtual void Serialize(const std::string& path) = 0;
		virtual void Deserialize(const std::string& path) = 0;

		std::vector<Object*>& GetDeserializedObjects();

	protected:
		FileId GetFileId(Object* object);
		Object* GetObjectRef(const FileId& fileId);

	protected:
		std::vector<Object*> m_ObjectsToSerialize;
		std::vector<Object*> m_DeserializedObjects;
		std::unordered_map<Object*, FileId> m_ObjectToFileId;
		std::unordered_map<FileId, Object*> m_FileIdToObject;
		FileId m_MaxId = 0;
	};
}