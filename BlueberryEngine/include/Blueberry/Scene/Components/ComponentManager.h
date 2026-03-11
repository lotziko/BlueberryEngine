#pragma once

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	struct ComponentIterator
	{
	public:
		ComponentIterator(Dictionary<ObjectId, Component*>* data) : m_Data(data) {}

		auto begin() { return m_Data->begin(); }
		auto end() { return m_Data->end(); }

	private:
		Dictionary<ObjectId, Component*>* m_Data;
	};

	using ComponentMap = Dictionary<ObjectId, Component*>;

	class ComponentManager
	{
	public:
		void AddComponent(Component* component);
		// Unsafe
		void AddComponent(Component* component, const TypeId& type);
		void RemoveComponent(Component* component);
		// Unsafe
		void RemoveComponent(Component* component, const TypeId& type);

		template<class ComponentType>
		ComponentIterator GetIterator();
		ComponentIterator GetIterator(const TypeId& type);
		ComponentMap& GetComponents(const TypeId& type);

	private:
		Dictionary<size_t, ComponentMap> m_Components;
	};

	template<class ComponentType>
	inline ComponentIterator ComponentManager::GetIterator()
	{
		return ComponentIterator(&m_Components[ComponentType::Type]);
	}

	inline void ComponentManager::AddComponent(Component* component)
	{
		m_Components[component->GetType()][component->GetObjectId()] = component;
	}

	inline void ComponentManager::AddComponent(Component* component, const TypeId& type)
	{
		m_Components[type][component->GetObjectId()] = component;
	}

	inline void ComponentManager::RemoveComponent(Component* component)
	{
		m_Components[component->GetType()].erase(component->GetObjectId());
	}

	inline void ComponentManager::RemoveComponent(Component* component, const TypeId& type)
	{
		m_Components[type].erase(component->GetObjectId());
	}

	inline ComponentIterator ComponentManager::GetIterator(const TypeId& type)
	{
		return ComponentIterator(&m_Components[type]);
	}

	inline ComponentMap& ComponentManager::GetComponents(const TypeId& type)
	{
		return m_Components[type];
	}
}