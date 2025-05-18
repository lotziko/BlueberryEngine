#include "Blueberry\Scene\Entity.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Core\ClassDB.h"
#include "..\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Entity, Object)
	{
		DEFINE_BASE_FIELDS(Entity, Object)
		DEFINE_FIELD(Entity, m_Components, BindingType::ObjectPtrArray, FieldOptions().SetObjectType(Component::Type))
		DEFINE_FIELD(Entity, m_IsActive, BindingType::Bool, {})
	}

	Entity::Entity(const String& name)
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