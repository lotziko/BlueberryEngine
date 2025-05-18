#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class ObjectInspector;

	class ObjectInspectorDB
	{
	public:
		static Dictionary<size_t, ObjectInspector*>& GetInspectors();
		static ObjectInspector* GetInspector(const size_t& type);
		static void Register(const size_t& type, ObjectInspector* inspector);
	private:
		static Dictionary<size_t, ObjectInspector*> s_Inspectors;
	};

	#define REGISTER_OBJECT_INSPECTOR( inspectorType, objectType ) ObjectInspectorDB::Register(TO_OBJECT_TYPE(TO_STRING(objectType)), new inspectorType());
}