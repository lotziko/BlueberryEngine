#include "bbpch.h"
#include "ObjectInspectorDB.h"

#include "ObjectInspector.h"

namespace Blueberry
{
	std::map<std::size_t, ObjectInspector*> ObjectInspectorDB::s_Inspectors = std::map<std::size_t, ObjectInspector*>();

	std::map<std::size_t, ObjectInspector*>& Blueberry::ObjectInspectorDB::GetInspectors()
	{
		return s_Inspectors;
	}

	ObjectInspector* ObjectInspectorDB::GetInspector(const std::size_t& id)
	{
		return s_Inspectors.find(id)->second;
	}

	void ObjectInspectorDB::Register(const std::size_t& id, ObjectInspector* inspector)
	{
		if (s_Inspectors.count(id) == 0)
		{
			s_Inspectors.insert({ id, inspector });
		}
	}
}