#pragma once

#include <map>
#include "Blueberry/Scene/Entity.h"

namespace Blueberry
{
	struct ComponentIterator
	{
	public:
		ComponentIterator(std::map<std::size_t, Component*> data) : m_Data(data) {}

		auto begin() { return m_Data.begin(); }
		auto end() { return m_Data.end(); }

	private:
		std::map<std::size_t, Component*> m_Data;
	};

	class ComponentManager
	{
	public:
		void AddComponent(Component* component);
		void RemoveComponent(Component* component);

		template<class ComponentType>
		ComponentIterator GetIterator();

	private:
		std::map<std::size_t, std::map<std::size_t, Component*>> m_Components;
	};

	template<class ComponentType>
	inline ComponentIterator ComponentManager::GetIterator()
	{
		return ComponentIterator(m_Components[ComponentType::Type]);
	}

	inline void ComponentManager::AddComponent(Component* component)
	{
		m_Components[component->GetType()][component->GetObjectId()] = component;
	}

	inline void ComponentManager::RemoveComponent(Component* component)
	{
		m_Components[component->GetType()].erase(component->GetObjectId());
	}
}