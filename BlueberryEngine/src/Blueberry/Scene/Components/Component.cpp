#include "bbpch.h"
#include "Component.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Component)

	Entity* Component::GetEntity()
	{
		return m_Entity.Get();
	}

	void Component::BindProperties()
	{
	}
}