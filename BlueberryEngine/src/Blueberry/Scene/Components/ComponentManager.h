#pragma once

#include <unordered_map>
#include "Blueberry/Scene/Entity.h"

namespace Blueberry
{
	struct ComponentIterator
	{
	public:
		ComponentIterator(std::unordered_map<ObjectId, Component*>* data) : m_Data(data) {}

		auto begin() { return m_Data->begin(); }
		auto end() { return m_Data->end(); }

	private:
		std::unordered_map<ObjectId, Component*>* m_Data;
	};

	using ComponentMap = std::unordered_map<ObjectId, Component*>;

	class ComponentManager
	{
	public:
		void AddComponent(Component* component);
		// Unsafe
		void AddComponent(Component* component, const size_t& type);
		void RemoveComponent(Component* component);
		// Unsafe
		void RemoveComponent(Component* component, const size_t& type);

		template<class ComponentType>
		ComponentIterator GetIterator();
		ComponentIterator GetIterator(const size_t& type);
		ComponentMap& GetComponents(const size_t& type);

	private:
		std::map<size_t, ComponentMap> m_Components;
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

	inline void ComponentManager::AddComponent(Component* component, const size_t& type)
	{
		m_Components[type][component->GetObjectId()] = component;
	}

	inline void ComponentManager::RemoveComponent(Component* component)
	{
		m_Components[component->GetType()].erase(component->GetObjectId());
	}

	inline void ComponentManager::RemoveComponent(Component* component, const size_t& type)
	{
		m_Components[type].erase(component->GetObjectId());
	}

	inline ComponentIterator ComponentManager::GetIterator(const size_t& type)
	{
		return ComponentIterator(&m_Components[type]);
	}

	inline ComponentMap& ComponentManager::GetComponents(const size_t& type)
	{
		return m_Components[type];
	}
}