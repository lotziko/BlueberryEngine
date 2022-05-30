#pragma once

#include "Blueberry\Core\Object.h"

class Entity;

class ComponentDefinitions
{
public:
	static std::vector<std::string> s_Names;
	ComponentDefinitions(std::string name) { s_Names.push_back(name); }
};

#define COMPONENT_DEFINITION(cls) static ComponentDefinitions myclass_##cls(#cls);

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