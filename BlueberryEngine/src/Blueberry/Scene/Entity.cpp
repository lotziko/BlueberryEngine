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

	void Entity::OnCreate()
	{
		for (auto&& componentSlot : m_Components)
		{
			componentSlot->OnCreate();
		}
	}

	void Entity::OnDestroy()
	{
		for (auto && componentSlot : m_Components)
		{
			if (componentSlot.IsValid())
			{
				//BB_INFO(componentSlot->GetTypeName() << " is destroyed.");
				//RemoveComponentFromScene(componentSlot.Get());
				componentSlot->OnDisable();
				componentSlot->OnDestroy();
				Object::Destroy(componentSlot.Get());
			}
		}
	}

	List<Component*> Entity::GetComponents()
	{
		List<Component*> components;
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
		if (m_Transform == nullptr)
		{
			m_Transform = static_cast<Transform*>(m_Components[0].Get());
		}
		return m_Transform;
	}

	Scene* Entity::GetScene()
	{
		return m_Scene;
	}

	const bool& Entity::IsActive()
	{
		return m_IsActive;
	}

	void Entity::SetActive(const bool& active)
	{
		if (active != m_IsActive)
		{
			m_IsActive = active;
			UpdateHierarchy(active);
		}
	}

	bool Entity::IsActiveInHierarchy()
	{
		if (m_IsActiveInHierarchy == -1)
		{
			Transform* parent = GetTransform()->GetParent();
			if (parent == nullptr)
			{
				m_IsActiveInHierarchy = m_IsActive;
				return m_IsActiveInHierarchy;
			}
			m_IsActiveInHierarchy = parent->GetEntity()->IsActiveInHierarchy();
			return m_IsActiveInHierarchy;
		}
		else
		{
			return m_IsActiveInHierarchy == 1;
		}
	}

	void Entity::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Entity)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &Entity::m_Name, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_Components), &Entity::m_Components, BindingType::ObjectPtrArray).SetObjectType(Component::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_IsActive), &Entity::m_IsActive, BindingType::Bool))
		END_OBJECT_BINDING()
	}

	void Entity::AddToCreatedComponents(Component* component)
	{
		if (m_Scene == nullptr)
		{
			return;
		}
		m_Scene->m_CreatedComponents.emplace_back(component);
	}

	void Entity::AddComponentToScene(Component* component, const size_t& type)
	{
		if (m_Scene == nullptr)
		{
			return;
		}
		m_Scene->m_ComponentManager.AddComponent(component, type);
	}

	void Entity::RemoveComponentFromScene(Component* component, const size_t& type)
	{
		if (m_Scene == nullptr)
		{
			return;
		}
		m_Scene->m_ComponentManager.RemoveComponent(component, type);
	}

	void Entity::UpdateHierarchy(const bool& active)
	{
		m_IsActiveInHierarchy = active;
		if (GetTransform()->GetChildrenCount() > 0)
		{
			for (auto child : GetTransform()->GetChildren())
			{
				child->GetEntity()->UpdateHierarchy(active);
			}
		}
		UpdateComponents();
	}

	void Entity::UpdateComponents()
	{
		if (m_IsActiveInHierarchy)
		{
			for (auto&& componentSlot : m_Components)
			{
				Component* component = componentSlot.Get();
				if (!component->m_IsActive)
				{
					component->m_IsActive = true;
					component->OnEnable();
				}
			}
		}
		else
		{
			for (auto&& componentSlot : m_Components)
			{
				Component* component = componentSlot.Get();
				if (component->m_IsActive)
				{
					component->m_IsActive = false;
					component->OnDisable();
				}
			}
		}
	}
}