#include "Blueberry\Scene\Components\Component.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Object)
	{
		DEFINE_BASE_FIELDS(Component, Object)
		DEFINE_FIELD(Component, m_Entity, BindingType::ObjectPtr, FieldOptions().SetObjectType(Entity::Type).SetVisibility(VisibilityType::Hidden))
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

	const size_t UpdatableComponent::Type = TO_OBJECT_TYPE(TO_STRING(UpdatableComponent));
}