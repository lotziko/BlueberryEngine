#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Scene\EnityComponent.h"
#include <map>

namespace Blueberry
{
	class ObjectInspector
	{
	public:
		virtual void Draw(Object* object) = 0;
	};

	class ObjectInspectors
	{
	public:
		static std::map<std::size_t, ObjectInspector*>& GetInspectors()
		{
			static std::map<std::size_t, ObjectInspector*> s_Inspectors = std::map<std::size_t, ObjectInspector*>();
			return s_Inspectors;
		}

		static ObjectInspector* GetInspector(const std::size_t& type)
		{
			std::map<std::size_t, ObjectInspector*>& inspectors = GetInspectors();
			if (inspectors.count(type) > 0)
			{
				return inspectors.find(type)->second;
			}
			return nullptr;
		}

		ObjectInspectors(const std::size_t& type, ObjectInspector* inspector) { ObjectInspectors::GetInspectors().insert({ type, inspector }); }
	};

#define OBJECT_INSPECTOR_DECLARATION(inspectorType, objectType) static ObjectInspectors ObjectInspector_##inspectorType(std::hash<std::string>()(TO_STRING(objectType)), new inspectorType());
}