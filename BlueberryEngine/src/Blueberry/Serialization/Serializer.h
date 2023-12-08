#pragma once

#include <rapidyaml\ryml.h>
#include <concurrent_vector.h>

namespace Blueberry
{
	using FileId = uint64_t;

	class Serializer
	{
	public:
		Serializer(const ryml::NodeRef& root);
		~Serializer() { }

		const ryml::NodeRef& GetRoot();

		void AddObject(Object* object);
		void AddObject(ryml::ConstNodeRef& node);
		void Serialize();
		void Deserialize();

		std::vector<Ref<Object>>& GetDeserializedObjects();

	private:
		void Serialize(Object* object);
		void Deserialize(Ref<Object>& object, int nodeIndex);
		FileId GetFileId(Object* object);
		Ref<Object> GetObjectRef(const FileId& fileId);

	private:
		ryml::NodeRef m_Root;
		std::vector<Object*> m_ObjectsToSerialize;
		std::vector<Ref<Object>> m_DeserializedObjects;
		std::unordered_map<Object*, FileId> m_ObjectToFileId;
		std::unordered_map<FileId, std::pair<Ref<Object>, int>> m_FileIdToObject;
		concurrency::concurrent_vector<std::string> m_Anchors;
		FileId m_MaxId = 0;
	};
}