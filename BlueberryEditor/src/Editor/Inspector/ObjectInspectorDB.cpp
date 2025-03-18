#include "bbpch.h"
#include "ObjectInspectorDB.h"
#include "ObjectInspector.h"

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	Dictionary<std::size_t, ObjectInspector*> ObjectInspectorDB::s_Inspectors = {};

	Dictionary<std::size_t, ObjectInspector*>& Blueberry::ObjectInspectorDB::GetInspectors()
	{
		return s_Inspectors;
	}

	ObjectInspector* ObjectInspectorDB::GetInspector(const std::size_t& id)
	{
		std::size_t inheritsId = id;
		while (true)
		{
			auto inspectorIt = s_Inspectors.find(inheritsId);
			if (inspectorIt != s_Inspectors.end())
			{
				return inspectorIt->second;
			}
			inheritsId = ClassDB::GetInfo(inheritsId).parentId;
			if (inheritsId == 0)
			{
				break;
			}
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