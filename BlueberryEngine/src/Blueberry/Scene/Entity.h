#pragma once

#include "Blueberry\Core\Object.h"

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

		Entity(const std::string& name)
		{
			SetName(name);
		}

		template<class ComponentType, typename... Args>
		void AddComponent(Args&&... params);

		template<class ComponentType>
		void AddComponent(Ref<ComponentType> component);

		template<class ComponentType>
		ComponentType* GetComponent();

		std::vector<Ref<Component>> GetComponents();

		template<class ComponentType>
		bool HasComponent();

		template<class ComponentType>
		void RemoveComponent(Ref<ComponentType> component);

		inline std::size_t GetId();

		virtual std::string ToString() const final;

		Transform* GetTransform();
		Scene* GetScene();

		static void BindProperties();

	private:
		void AddComponentIntoScene(Component* component);
		void RemoveComponentFromScene(Component* component);

		void Destroy();

	private:
		std::vector<Ref<Component>> m_Components;

		std::size_t m_Id;

		Transform* m_Transform;
		Scene* m_Scene;

		friend class Scene;
	};

	template<class ComponentType, typename... Args>
	inline void Entity::AddComponent(Args&&... params)
	{
		static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");

		auto& componentToAdd = ObjectDB::CreateObject<ComponentType>(std::forward<Args>(params)...);

		int index = 0;
		for (auto && componentSlot : m_Components)
		{
			if (componentSlot == nullptr)
			{
				componentSlot = std::move(componentToAdd);
				break;
			}
			++index;
		}

		componentToAdd->m_Entity = this;
		AddComponentIntoScene(componentToAdd.get());
		if (index >= m_Components.size())
		{
			m_Components.emplace_back(std::move(componentToAdd));
		}
	}

	template<class ComponentType>
	inline void Entity::AddComponent(Ref<ComponentType> component)
	{
		int index = 0;
		for (auto && componentSlot : m_Components)
		{
			if (componentSlot == nullptr)
			{
				componentSlot = std::move(component);
				break;
			}
			++index;
		}

		component->m_Entity = this;
		AddComponentIntoScene(component.get());
		if (index >= m_Components.size())
		{
			m_Components.emplace_back(std::move(component));
		}
	}

	template<class ComponentType>
	inline ComponentType* Entity::GetComponent()
	{
		for (auto && component : m_Components)
		{
			if (component->IsClassType(ComponentType::Type))
			{
				return dynamic_cast<ComponentType*>(component.get());
			}
		}

		return Ref<ComponentType>().get();
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
	inline void Entity::RemoveComponent(Ref<ComponentType> component)
	{
		RemoveComponentFromScene(component.get());
		auto& index = std::find(m_Components.begin(), m_Components.end(), component);
		m_Components.erase(index);
		ObjectDB::DestroyObject(component.get());
	}
}