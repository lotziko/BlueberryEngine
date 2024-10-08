#pragma once

#include "Blueberry\Core\Object.h"

class Component;
class Transform;
class Scene;

class Entity : Object
{
public:
	Entity() = default;
	
	Entity(const std::string& name) : m_Name(name)
	{
	}

	template<class ComponentType, typename... Args>
	void AddComponent(Args&&... params);

	template<class ComponentType>
	ComponentType* GetComponent();

	template<class ComponentType>
	bool HasComponent();
	
	inline std::size_t GetId() { return m_Id; }

	virtual std::string ToString() const final { return m_Name; }

	inline Transform* GetTransform() { return m_Transform; }

	void Destroy();
private:
	void AddComponentIntoScene(Component* component);
	void RemoveComponentFromScene(Component* component);
	
private:
	std::vector<Ref<Component>> m_Components;

	std::size_t m_Id;
	std::string m_Name;

	Transform* m_Transform;
	Scene* m_Scene;

	friend class Scene;
};

template<class ComponentType, typename... Args>
inline void Entity::AddComponent(Args&&... params)
{
	static_assert(std::is_base_of<Component, ComponentType>::value, "Type is not derived from Component.");

	auto& componentToAdd = CreateRef<ComponentType>(std::forward<Args>(params)...);

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

inline void Entity::Destroy()
{
	for (auto && componentSlot : m_Components)
	{
		RemoveComponentFromScene(componentSlot.get());
	}
}