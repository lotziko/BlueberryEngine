#pragma once

#include "Blueberry\Core\Object.h"

class Entity;

class ComponentDefinitions
{
public:
	static std::vector<std::string>& GetDefinitions()
	{
		static std::vector<std::string> s_Definitions = std::vector<std::string>();
		return s_Definitions;
	}

	ComponentDefinitions(std::string name) { ComponentDefinitions::GetDefinitions().push_back(name); }
};

#define COMPONENT_DEFINITION(componentType) static ComponentDefinitions ComponentDefinition_##componentType(#componentType);

class Component : public Object
{
	OBJECT_DECLARATION(Component)
public:
	virtual ~Component() = default;

	inline Entity* GetEntity() { return m_Entity; }

private:
	Entity* m_Entity;

	friend class Entity;
};