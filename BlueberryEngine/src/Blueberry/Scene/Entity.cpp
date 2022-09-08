#include "bbpch.h"
#include "Entity.h"

#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Entity)

	std::vector<Ref<Component>> Entity::GetComponents()
	{
		return m_Components;
	}

	std::size_t Entity::GetId()
	{
		return m_Id;
	}

	std::string Entity::ToString() const
	{
		return m_Name;
	}

	Transform* Entity::GetTransform()
	{
		return m_Transform;
	}

	Scene* Entity::GetScene()
	{
		return m_Scene;
	}

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