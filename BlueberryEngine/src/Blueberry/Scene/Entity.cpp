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
		BIND_FIELD(FieldInfo(TO_STRING(m_Components), &Entity::m_Components, BindingType::ObjectPtrArray).SetObjectType(Component::Type))
		END_OBJECT_BINDING()
	}

	void Entity::OnDestroy()
	{
		for (auto && componentSlot : m_Components)
		{
			if (componentSlot.IsValid())
			{
				//BB_INFO(componentSlot->GetTypeName() << " is destroyed.");
				RemoveComponentFromScene(componentSlot.Get());
				componentSlot->OnDestroy();
				Object::Destroy(componentSlot.Get());
			}
		}
	}

	void Entity::AddComponentIntoScene(Component* component)
	{
		if (m_Scene == nullptr)
		{
			return;
		}

		std::size_t type = component->GetType();
		m_Scene->m_ComponentManager.AddComponent(component);
		m_Scene->m_CreatedComponents.emplace_back(component);
	}

	void Entity::RemoveComponentFromScene(Component* component)
	{
		if (m_Scene == nullptr)
		{
			return;
		}

		std::size_t type = component->GetType();
		m_Scene->m_ComponentManager.RemoveComponent(component);
	}
}