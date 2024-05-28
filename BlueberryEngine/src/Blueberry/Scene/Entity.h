#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Component;
	class Transform;
	class Scene;

	class Entity : public Object
	{
		OBJECT_DECLARATION(Entity)
	public:
		Entity() = default;

		Entity(const std::string& name);

		template<class ComponentType>
		void AddComponent();

		template<class ComponentType>
		void AddComponent(ComponentType* component);

		template<class ComponentType>
		ComponentType* GetComponent();

		std::vector<Component*> GetComponents();

		template<class ComponentType>
		bool HasComponent();

		template<class ComponentType>
		void RemoveComponent(ComponentType* component);

		Transform* GetTransform();
		Scene* GetScene();

		static void BindProperties();

		virtual void OnDestroy() override;

	private:
		void AddComponentIntoScene(Component* component);
		void RemoveComponentFromScene(Component* component);

	private:
		std::vector<ObjectPtr<Component>> m_Components;

		ObjectPtr<Transform> m_Transform;
		Scene* m_Scene;

		friend class Scene;
	};

	template<class ComponentType>
	inline void Entity::AddComponent()
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");

		ComponentType* componentToAdd = Object::Create<ComponentType>();
		ObjectPtr<Component> weakPtr((Component*)componentToAdd);

		int index = 0;
		for (auto componentSlot : m_Components)
		{
			if (!componentSlot.IsValid())
			{
				componentSlot = std::move(weakPtr);
				break;
			}
			++index;
		}

		componentToAdd->m_Entity = ObjectPtr<Entity>(this);
		AddComponentIntoScene(componentToAdd);
		if (index >= m_Components.size())
		{
			m_Components.emplace_back(std::move(weakPtr));
		}
	}

	template<class ComponentType>
	inline void Entity::AddComponent(ComponentType* component)
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");

		ObjectPtr<Component> weakPtr((Component*)component);
		int index = 0;
		for (auto& componentSlot : m_Components)
		{
			if (!componentSlot.IsValid())
			{
				componentSlot = std::move(weakPtr);
				break;
			}
			++index;
		}

		component->m_Entity = ObjectPtr<Entity>(this);
		AddComponentIntoScene(component);
		if (index >= m_Components.size())
		{
			m_Components.emplace_back(std::move(weakPtr));
		}
	}

	template<class ComponentType>
	inline ComponentType* Entity::GetComponent()
	{
		for (auto& component : m_Components)
		{
			if (component->IsClassType(ComponentType::Type))
			{
				return (ComponentType*)component.Get();
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
		RemoveComponentFromScene(component);
		auto& index = std::find(m_Components.begin(), m_Components.end(), component);
		m_Components.erase(index);
		Object::Destroy(component);
	}
}