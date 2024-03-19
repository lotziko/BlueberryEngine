#include "bbpch.h"
#include "Entity.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Entity)

	Entity::Entity(const std::string& name)
	{
		SetName(name);
	}

	std::vector<Component*> Entity::GetComponents()
	{
		std::vector<Component*> components;
		for (auto component : m_Components)
		{
			if (component.IsValid())
			{
				components.emplace_back(component.Get());
			}
		}
		return components;
	}

	std::size_t Entity::GetId()
	{
		return m_Id;
	}

	Transform* Entity::GetTransform()
	{
		return m_Transform.Get();
	}

	Scene* Entity::GetScene()
	{
		return m_Scene;
	}

	void Entity::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Entity)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &Entity::m_Name, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_Components), &Entity::m_Components, BindingType::ObjectPtrArray, Component::Type))
		END_OBJECT_BINDING()
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

	void Entity::Destroy()
	{
		for (auto && componentSlot : m_Components)
		{
			BB_INFO(componentSlot->GetTypeName() << " is destroyed.");
			RemoveComponentFromScene(componentSlot.Get());
			componentSlot->Destroy();
			Object::Destroy(componentSlot.Get());
		}
	}
}