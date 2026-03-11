#include "Blueberry\Scene\Entity.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Entity, Object)
	{
		DEFINE_BASE_FIELDS(Entity, Object)
		DEFINE_FIELD(Entity, m_Components, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&Component::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(Entity, m_IsActive, BindingType::Bool, FieldOptions().SetUpdateCallback(MethodBind::Create(&Entity::UpdateHierarchy)))
	}

	Entity::Entity(const String& name)
	{
		SetName(name);
	}

	void Entity::OnCreate()
	{
	}

	void Entity::OnDestroy()
	{
		for (auto& componentSlot : m_Components)
		{
			if (componentSlot.IsValid())
			{
				Component* component = componentSlot.Get();
				RemoveComponentFromScene(component);
				if (component->CanExecute())
				{
					if (component->m_IsActive)
					{
						component->OnDisable();
						component->m_IsActive = false;
					}
					component->OnDestroy();
				}
				Object::Destroy(component);
			}
		}
		m_Components.clear();
	}

	void Entity::AddComponent(Component* component)
	{
		int index = 0;
		for (auto& componentSlot : m_Components)
		{
			if (!componentSlot.IsValid())
			{
				componentSlot = component;
				break;
			}
			++index;
		}

		component->m_Entity = this;
		if (index >= m_Components.size())
		{
			m_Components.push_back(component);
		}
		if (IsActiveInHierarchy())
		{
			AddComponentToScene(component);
			if (component->CanExecute())
			{
				if (!component->m_IsCreated)
				{
					component->OnCreate();
					component->m_IsCreated = true;
				}
				if (!component->m_IsActive)
				{
					component->OnEnable();
					component->m_IsActive = true;
				}
			}
		}
	}

	Component* Entity::GetComponentAt(const size_t& index)
	{
		if (index >= 0 && index < m_Components.size())
		{
			return m_Components[index].Get();
		}
		return nullptr;
	}

	const size_t Entity::GetComponentCount()
	{
		return m_Components.size();
	}

	void Entity::RemoveComponent(Component* component)
	{
		if (component->m_Entity == this)
		{
			RemoveComponentFromScene(component);
			if (component->CanExecute())
			{
				component->OnDisable();
				component->OnDestroy();
			}
			auto& index = std::find(m_Components.begin(), m_Components.end(), component);
			m_Components.erase(index);
			Object::Destroy(component);
		}
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
		return m_IsActiveInHierarchy;
	}

	void Entity::UpdateHierarchy()
	{
		if (m_Scene == nullptr)
		{
			return;
		}
		Transform* parent = GetTransform()->GetParent();
		if (parent != nullptr)
		{
			UpdateHierarchy(parent->GetEntity()->m_IsActiveInHierarchy);
		}
		else
		{
			UpdateHierarchy(true);
		}
	}

	bool Entity::HasComponent(const TypeId& type)
	{
		for (auto& component : m_Components)
		{
			if (component.IsValid() && component->IsClassType(type))
			{
				return true;
			}
		}
		return false;
	}

	Component* Entity::GetComponent(const TypeId& type)
	{
		for (auto& component : m_Components)
		{
			if (component.IsValid() && component->IsClassType(type))
			{
				return component.Get();
			}
		}
		return nullptr;
	}

	Component* Entity::GetComponentInParent(const TypeId& type)
	{
		Entity* entity = this;
		while (entity != nullptr)
		{
			for (auto& component : entity->m_Components)
			{
				if (component.IsValid() && component->IsClassType(type))
				{
					return component.Get();
				}
			}
			Transform* parent = entity->GetTransform()->GetParent();
			if (parent == nullptr)
			{
				break;
			}
			entity = parent->GetEntity();
		}
		return nullptr;
	}

	Component* Entity::GetComponentInChildren(const TypeId& type)
	{
		List<Entity*> stack;
		stack.push_back(this);
		while (stack.size() > 0)
		{
			Entity* entity = stack[stack.size() - 1];
			Component* component = entity->GetComponent(type);
			if (component != nullptr)
			{
				return component;
			}
			stack.pop_back();
			for (auto& child : entity->GetTransform()->GetChildren())
			{
				stack.push_back(child.Get()->GetEntity());
			}
		}
		return nullptr;
	}

	void Entity::AddComponentToScene(Component* component)
	{
		if (m_Scene == nullptr)
		{
			return;
		}
		for (TypeId type : ClassDB::GetInfo(component->GetType())->iterators)
		{
			m_Scene->m_ComponentManager.AddComponent(component, type);
		}
	}

	void Entity::RemoveComponentFromScene(Component* component)
	{
		if (m_Scene == nullptr)
		{
			return;
		}
		for (TypeId type : ClassDB::GetInfo(component->GetType())->iterators)
		{
			m_Scene->m_ComponentManager.RemoveComponent(component, type);
		}
	}

	void Entity::UpdateHierarchy(const bool& active)
	{
		bool newActive = m_IsActive & active;
		if (m_IsActiveInHierarchy != newActive)
		{
			m_IsActiveInHierarchy = newActive;
			UpdateComponents();
		}
		if (GetTransform()->GetChildrenCount() > 0)
		{
			for (auto& child : GetTransform()->GetChildren())
			{
				child.Get()->GetEntity()->UpdateHierarchy(newActive);
			}
		}
	}

	void Entity::UpdateComponents()
	{
		if (m_IsActiveInHierarchy)
		{
			EnableComponents();
		}
		else
		{
			DisableComponents();
		}
	}

	void Entity::EnableComponents()
	{
		for (auto&& componentSlot : m_Components)
		{
			Component* component = componentSlot.Get();
			AddComponentToScene(component);
			if (component->CanExecute())
			{
				if (!component->m_IsCreated)
				{
					component->OnCreate();
					component->m_IsCreated = true;
				}
				if (!component->m_IsActive)
				{
					component->OnEnable();
					component->m_IsActive = true;
				}
			}
		}
	}

	void Entity::DisableComponents()
	{
		for (auto&& componentSlot : m_Components)
		{
			Component* component = componentSlot.Get();
			RemoveComponentFromScene(component);
			if (component->m_IsActive && component->CanExecute())
			{
				component->OnDisable();
				component->m_IsActive = false;
			}
		}
	}
}