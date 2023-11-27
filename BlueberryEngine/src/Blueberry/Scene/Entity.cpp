#include "bbpch.h"
#include "Entity.h"

#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Entity)

	void Entity::Serialize(SerializationContext& context, ryml::NodeRef& node)
	{
		ryml::NodeRef componentsNode = node["Components"];
		componentsNode |= ryml::MAP;
		for (auto&& componentSlot : m_Components)
		{
			ryml::NodeRef componentNode = componentsNode.append_child() << ryml::key(componentSlot->ToString());
			componentNode |= ryml::MAP;
			componentSlot->Serialize(context, componentNode);
		}
	}

	void Entity::Deserialize(SerializationContext& context, ryml::NodeRef& node)
	{

	}

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
		return m_Name;
	}

	Transform* Entity::GetTransform()
	{
		return m_Transform;
	}

	Scene* Entity::GetScene()
	{
		return m_Scene;
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
}