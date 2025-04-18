#include "bbpch.h"
#include "Component.h"

#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Object)
	{
		DEFINE_BASE_FIELDS(Component, Object)
		DEFINE_FIELD(Component, m_Entity, BindingType::ObjectPtr, FieldOptions().SetObjectType(Entity::Type).SetHidden())
	}

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