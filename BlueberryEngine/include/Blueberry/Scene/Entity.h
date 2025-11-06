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

		virtual void OnCreate() override;
		virtual void OnDestroy() override;

		template<class ComponentType>
		ComponentType* AddComponent();

		template<class ComponentType>
		void AddComponent(ComponentType* component);

		template<class ComponentType>
		ComponentType* GetComponent();

		Component* GetComponent(const uint32_t& index);

		const uint32_t GetComponentCount();

		template<class ComponentType>
		bool HasComponent();

		template<class ComponentType>
		void RemoveComponent(ComponentType* component);

		Transform* GetTransform();
		Scene* GetScene();
		
		const bool& IsActive();
		void SetActive(const bool& active);
		bool IsActiveInHierarchy();

	private:
		void AddToCreatedComponents(Component* component);
		void AddComponentToScene(Component* component, const size_t& type);
		void RemoveComponentFromScene(Component* component, const size_t& type);
		void UpdateHierarchy();
		void UpdateHierarchy(const bool& active);
		void UpdateComponents();

	private:
		List<ObjectPtr<Component>> m_Components;
		bool m_IsActive = true;

		Transform* m_Transform;
		Scene* m_Scene;
		int8_t m_IsActiveInHierarchy = -1;

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
		AddToCreatedComponents(componentToAdd);
		if (index >= m_Components.size())
		{
			m_Components.emplace_back(componentToAdd);
		}
		componentToAdd->OnCreate();
		if (IsActiveInHierarchy())
		{
			// TODO handle prefabs
			componentToAdd->OnEnable();
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
		AddToCreatedComponents(component);
		if (index >= m_Components.size())
		{
			m_Components.emplace_back(component);
		}
		component->OnCreate();
		if (IsActiveInHierarchy())
		{
			component->OnEnable();
		}
	}

	template<class ComponentType>
	inline ComponentType* Entity::GetComponent()
	{
		for (auto& component : m_Components)
		{
			if (component->IsClassType(ComponentType::Type))
			{
				return static_cast<ComponentType*>(component.Get());
			}
		}

		return nullptr;
	}

	template<class ComponentType>
	inline bool Entity::HasComponent()
	{
		for (int index = m_ComponentsTypeHash.First(ComponentType::Type); index != INVALID_ID; index = m_ComponentsTypeHash.Next(index))
		{
			if (m_Components[index]->IsClassType(ComponentType::Type))
			{
				return true;
			}
		}

		return false;
	}

	template<class ComponentType>
	inline void Entity::RemoveComponent(ComponentType* component)
	{
		//RemoveComponentFromScene(component);
		auto& index = std::find(m_Components.begin(), m_Components.end(), component);
		m_Components.erase(index);
		Object::Destroy(component);
	}
}