#include "bbpch.h"
#include "Component.h"

#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Component)

	Entity* Component::GetEntity()
	{
		return m_Entity.Get();
	}

	Transform* Component::GetTransform()
	{
		return m_Entity.Get()->GetTransform();
	}

	Scene* Component::GetScene()
	{
		return m_Entity.Get()->GetScene();
	}

	void Component::BindProperties()
	{
	}

	void Component::AddToSceneComponents(const size_t& type)
	{
		m_Entity.Get()->AddComponentToScene(this, type);
	}

	void Component::RemoveFromSceneComponents(const size_t& type)
	{
		m_Entity.Get()->RemoveComponentFromScene(this, type);
	}

	const size_t UpdatableComponent::Type = TO_OBJECT_TYPE(TO_STRING(UpdatableComponent));
}