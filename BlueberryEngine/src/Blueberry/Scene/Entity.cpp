#include "bbpch.h"
#include "Entity.h"

#include "Blueberry\Scene\Scene.h"

void Entity::AddComponentIntoScene(Component* component)
{
	std::size_t type = component->GetType();
	if (m_Scene->m_ComponentManagers.count(type))
	{
		m_Scene->m_ComponentManagers[type]->AddComponent(this, component);
	}
}

void Entity::RemoveComponentFromScene(Component* component)
{
	std::size_t type = component->GetType();
	if (m_Scene->m_ComponentManagers.count(type))
	{
		m_Scene->m_ComponentManagers[type]->RemoveComponent(this);
	}
}
