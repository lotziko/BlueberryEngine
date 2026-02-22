#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Component;
	class Transform;
	class Scene;

	class BB_API Entity : public Object
	{
		OBJECT_DECLARATION(Entity)
		
	public:
		Entity() = default;
		Entity(const String& name);

		void OnCreate();
		void OnDestroy();

		template<class ComponentType>
		ComponentType* AddComponent();

		template<class ComponentType>
		void AddComponent(ComponentType* component);

		template<class ComponentType>
		ComponentType* GetComponent();

		Component* GetComponent(const size_t& index);

		const size_t GetComponentCount();

		template<class ComponentType>
		bool HasComponent();

		template<class ComponentType>
		void RemoveComponent(ComponentType* component);

		Transform* GetTransform();
		Scene* GetScene();
		
		const bool& IsActive();
		void SetActive(const bool& active);
		bool IsActiveInHierarchy();

		void UpdateHierarchy();

	private:
		void AddComponentToScene(Component* component);
		void RemoveComponentFromScene(Component* component);
		void UpdateHierarchy(const bool& active);
		void UpdateComponents();
		void EnableComponents();
		void DisableComponents();

	private:
		List<ObjectPtr<Component>> m_Components;
		bool m_IsActive = true;

		Transform* m_Transform;
		Scene* m_Scene;
		bool m_IsActiveInHierarchy = false;

		friend class Scene;
		friend class Component;
		friend class Transform;
	};

	template<class ComponentType>
	inline ComponentType* Entity::AddComponent()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");

		ComponentType* componentToAdd = Object::Create<ComponentType>();

		int index = 0;
		for (auto& componentSlot : m_Components)
		{
			if (!componentSlot.IsValid())
			{
				componentSlot = componentToAdd;
				break;
			}
			++index;
		}

		componentToAdd->m_Entity = ObjectPtr<Entity>(this);
		if (index >= m_Components.size())
		{
			m_Components.push_back(componentToAdd);
		}
		if (IsActiveInHierarchy())
		{
			AddComponentToScene(componentToAdd);
			if (componentToAdd->CanExecute())
			{
				componentToAdd->OnCreate();
				componentToAdd->m_IsCreated = true;
				componentToAdd->OnEnable();
				componentToAdd->m_IsActive = true;
			}
		}
		return componentToAdd;
	}

	template<class ComponentType>
	inline void Entity::AddComponent(ComponentType* component)
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");

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

		component->m_Entity = ObjectPtr<Entity>(this);
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

	template<class ComponentType>
	inline ComponentType* Entity::GetComponent()
	{
		for (auto& component : m_Components)
		{
			if (component.IsValid() && component->IsClassType(ComponentType::Type))
			{
				return static_cast<ComponentType*>(component.Get());
			}
		}

		return nullptr;
	}

	template<class ComponentType>
	inline bool Entity::HasComponent()
	{
		for (auto& component : m_Components)
		{
			if (component.IsValid() && component->IsClassType(ComponentType::Type))
			{
				return true;
			}
		}
		return false;
	}

	template<class ComponentType>
	inline void Entity::RemoveComponent(ComponentType* component)
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