#include "bbpch.h"
#include "Entity.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\YamlHelper.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Entity)

	Entity::Entity(const std::string& name)
	{
		SetName(name);
	}

	std::vector<Ref<Component>>& Entity::GetComponents()
	{
		return m_Components;
	}

	std::size_t Entity::GetId()
	{
		return m_Id;
	}

	Ref<Transform>& Entity::GetTransform()
	{
		return m_Transform;
	}

	Scene* Entity::GetScene()
	{
		return m_Scene;
	}

	void Entity::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Entity)
		BIND_FIELD("m_Name", &Entity::m_Name, BindingType::String)
		BIND_FIELD("m_Components", &Entity::m_Components, BindingType::ObjectRefArray)
		END_OBJECT_BINDING()
	}

	void Entity::AddComponentIntoScene(Component* component)
	{
		std::size_t type = component->GetType();
		m_Scene->m_ComponentManager.AddComponent(this, component);
	}

	void Entity::RemoveComponentFromScene(Component* component)
	{
		std::size_t type = component->GetType();
		m_Scene->m_ComponentManager.RemoveComponent(this, component);
	}

	void Entity::Destroy()
	{
		for (auto && componentSlot : m_Components)
		{
			RemoveComponentFromScene(componentSlot.get());
			ObjectDB::DestroyObject(std::dynamic_pointer_cast<Object>(componentSlot));
		}
	}
}