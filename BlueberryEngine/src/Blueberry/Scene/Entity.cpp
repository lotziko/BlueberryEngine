#include "bbpch.h"
#include "Entity.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Serialization\YamlHelper.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Entity)

	std::vector<Ref<Component>> Entity::GetComponents()
	{
		return m_Components;
	}

	std::size_t Entity::GetId()
	{
		return m_Id;
	}

	std::string Entity::ToString() const
	{
		return "Entity";
	}

	Transform* Entity::GetTransform()
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