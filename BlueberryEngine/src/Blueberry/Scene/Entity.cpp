#include "bbpch.h"
#include "Entity.h"

#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Entity)

	void Entity::AddComponentIntoScene(Component* component)
	{
		std::size_t type = component->GetType();
		m_Scene->m_ComponentManager.AddComponent(this, component);
	}

	void Entity::RemoveComponentFromScene(Component* component)
	{
		std::size_t type = component->GetType();
		m_Scene->m_ComponentManager.RemoveComponent(this, component);
	}
}