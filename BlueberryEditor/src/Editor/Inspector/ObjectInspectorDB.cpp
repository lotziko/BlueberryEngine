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
		auto inspectorIt = s_Inspectors.find(id);
		if (inspectorIt != s_Inspectors.end())
		{
			return inspectorIt->second;
		}
		return nullptr;
	}

	void ObjectInspectorDB::Register(const std::size_t& id, ObjectInspector* inspector)
	{
		if (s_Inspectors.count(id) == 0)
		{
			s_Inspectors.insert({ id, inspector });
		}
	}
}