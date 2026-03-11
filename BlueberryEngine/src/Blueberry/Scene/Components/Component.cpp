#include "Blueberry\Scene\Components\Component.h"

#include "Blueberry\Core\Application.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Object)
	{
		DEFINE_BASE_FIELDS(Component, Object)
		DEFINE_FIELD(Component, m_Entity, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Entity::Type).SetVisibility(VisibilityType::Hidden))
	}

	const String& Component::GetName()
	{
		return m_Entity->GetName();
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

	const bool& Component::IsActive()
	{
		return m_IsActive;
	}

	bool Component::CanExecute()
	{
		return Application::IsRunning() || ClassDB::GetInfo(GetType())->executeAlways;
	}

	TypeId UpdatableComponent::Type = 0;
	const String UpdatableComponent::TypeName = "UpdatableComponent";
}