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

		void Serialize(const Ref<Object>& object);

	private:
		FileId GetFileId(Object* object);

	private:
		ryml::NodeRef m_Root;
		std::map<Object*, FileId> m_ObjectToFileId;
		std::map<FileId, ObjectId> m_FileIdToObjectId;
		concurrency::concurrent_vector<std::string> m_Anchors;
		FileId m_MaxId = 0;
	};
}