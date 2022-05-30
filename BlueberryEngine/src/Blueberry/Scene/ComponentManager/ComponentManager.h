#pragma once

#include <map>
#include "Blueberry/Scene/Entity.h"

class ComponentManager
{
public:
	virtual void AddComponent(Entity* entity, Component* component) = 0;
	virtual void RemoveComponent(Entity* entity) = 0;
};

template<class ComponentType>
struct ComponentIterator
{
public:
	ComponentIterator(std::map<std::size_t, ComponentType*> data) : m_Data(data) {}

	auto begin() { return m_Data.begin(); }
	auto end() { return m_Data.end(); }

private:
	std::map<std::size_t, ComponentType*> m_Data;
};

template<class ComponentType>
class TComponentManager : public ComponentManager
{
public:
	virtual void AddComponent(Entity* entity, Component* component) final;
	virtual void RemoveComponent(Entity* entity) final;

	ComponentIterator<ComponentType> GetIterator();
	
	static std::size_t GetComponentType();

protected:
	std::map<std::size_t, ComponentType*> m_Components;
};

template<class ComponentType>
inline void TComponentManager<ComponentType>::AddComponent(Entity* entity, Component* component)
{
	m_Components[entity->GetId()] = static_cast<ComponentType*>(component);
}

template<class ComponentType>
inline void TComponentManager<ComponentType>::RemoveComponent(Entity* entity)
{
	m_Components.erase(entity->GetId());
}

template<class ComponentType>
inline ComponentIterator<ComponentType> TComponentManager<ComponentType>::GetIterator()
{
	return ComponentIterator<ComponentType>(m_Components);
}

template<class ComponentType>
inline std::size_t TComponentManager<ComponentType>::GetComponentType()
{
	return ComponentType::Type;
}
