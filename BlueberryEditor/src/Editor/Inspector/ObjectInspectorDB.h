#pragma once

#include <string>

namespace Blueberry
{
	class ObjectInspector;

	class ObjectInspectorDB
	{
	public:
		static std::map<std::size_t, ObjectInspector*>& GetInspectors();
		static ObjectInspector* GetInspector(const std::size_t& type);
		static void Register(const std::size_t& type, ObjectInspector* inspector);
	private:
		static std::map<std::size_t, ObjectInspector*> s_Inspectors;
	};

	#define REGISTER_OBJECT_INSPECTOR( inspectorType, objectType ) ObjectInspectorDB::Register(TO_OBJECT_TYPE(TO_STRING(objectType)), new inspectorType());
}